"""Portable four-angle PolarDehaze V3 inference."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from poldehaze_v2.models import PolarDehazeNet
from poldehaze_v2.data.polar_features import compute_physics_input

ANGLES = ('0', '45', '90', '135')


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = PolarDehazeNet.from_config(payload['config']).to(device).eval()
    model.load_state_dict(payload['model'], strict=True)
    return model, payload['config']


def read_angles(paths, input_scale):
    arrays = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image).copy()
            if array.ndim != 2:
                raise ValueError(f'{path}: expected one grayscale image per analyzer angle')
            if input_scale is not None:
                scale = input_scale
            elif array.dtype == np.uint8:
                scale = 255.0
            elif array.dtype == np.uint16 or image.mode in ('I', 'I;16', 'I;16B', 'I;16L'):
                scale = 65535.0
            elif np.issubdtype(array.dtype, np.floating):
                scale = 1.0
            else:
                raise ValueError(f'{path}: unsupported bit depth; supply --input-scale')
        arrays.append(array.astype(np.float32) / scale)
    if len({a.shape for a in arrays}) != 1:
        raise ValueError('The four angle images must have identical dimensions')
    raw = np.stack(arrays)
    if not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1:
        raise ValueError('Scaled input must be finite and within [0,1]')
    return torch.from_numpy(raw)[None]


def forward(model, config, raw):
    data = config['data']
    phys = compute_physics_input(raw, data['phys_channels'],
                                 normalize_s0=data['phys_normalize_s0'])
    pred = model(raw, phys)
    return torch.cat((pred['s0'], pred['s12']), dim=1)


@torch.inference_mode()
def restore(model, config, raw, mode='whole'):
    height, width = raw.shape[-2:]
    if mode == 'whole':
        padded = F.pad(raw, (0, (-width) % 16, 0, (-height) % 16), mode='replicate')
        return forward(model, config, padded)[..., :height, :width]
    # Blend linear Stokes fields; derive DoLP/AoLP after stitching.
    margin, tile, stride = 64, 256, 128
    padded = F.pad(raw, (margin, margin, margin, margin), mode='reflect')
    h, w = padded.shape[-2:]
    padded = F.pad(padded, (0, max(0, tile-w), 0, max(0, tile-h)), mode='replicate')
    h, w = padded.shape[-2:]
    ys = sorted(set(list(range(0, h-tile+1, stride)) + [h-tile]))
    xs = sorted(set(list(range(0, w-tile+1, stride)) + [w-tile]))
    hann = torch.hann_window(tile, periodic=False, device=raw.device)
    weight = (hann[:, None] * hann[None, :])[None, None]
    result = torch.zeros((1, 3, h, w), device=raw.device)
    weights = torch.zeros((1, 1, h, w), device=raw.device)
    for y in ys:
        for x in xs:
            result[..., y:y+tile, x:x+tile] += forward(model, config, padded[..., y:y+tile, x:x+tile]) * weight
            weights[..., y:y+tile, x:x+tile] += weight
    crop = (..., slice(margin, margin+height), slice(margin, margin+width))
    return result[crop] / weights[crop]


def save_result(stokes, folder):
    folder.mkdir(parents=True, exist_ok=True)
    s0, s1, s2 = stokes[0].float().cpu().numpy()
    dolp = np.clip(np.hypot(s1, s2) / np.maximum(s0, 1e-8), 0, 1)
    aolp = np.mod(0.5 * np.arctan2(s2, s1), np.pi)
    np.savez_compressed(folder / 'stokes.npz', s0=s0, s1=s1, s2=s2, dolp=dolp, aolp_rad=aolp)
    for name, value in [('s0', s0), ('dolp', dolp), ('aolp', aolp / np.pi)]:
        Image.fromarray(np.rint(np.clip(value, 0, 1)*65535).astype(np.uint16)).save(folder / f'{name}_16bit.png')
    Image.fromarray(np.rint(np.clip(s0, 0, 1)*255).astype(np.uint8)).save(folder / 's0_preview.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='Folder containing 0/45/90/135 subfolders with matching filenames')
    source.add_argument('--angles', nargs=4, type=Path, metavar=('I0', 'I45', 'I90', 'I135'))
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'weights/poldehazeV3_final.pt')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mode', choices=('whole', 'sliding'), default='whole')
    parser.add_argument('--input-scale', type=float, help='Explicit common divisor; default uint8=255, uint16=65535, float=1')
    args = parser.parse_args()
    if args.input_scale is not None and args.input_scale <= 0:
        parser.error('--input-scale must be positive')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, config = load_model(args.checkpoint, args.device)
    if args.input:
        names = sorted(p.name for p in (args.input / '0').iterdir()
                       if p.suffix.lower() in ('.png', '.tif', '.tiff', '.bmp'))
        if not names:
            parser.error('No supported images found under the 0 subfolder')
        jobs = [(Path(name).stem, [args.input / angle / name for angle in ANGLES]) for name in names]
    else:
        jobs = [(args.angles[0].stem, args.angles)]
    args.output.mkdir(parents=True, exist_ok=True)
    for index, (name, paths) in enumerate(jobs, 1):
        raw = read_angles(paths, args.input_scale).to(args.device)
        if args.mode == 'sliding' and min(raw.shape[-2:]) <= 64:
            parser.error('Sliding mode requires image height and width greater than 64')
        output = restore(model, config, raw, args.mode)
        save_result(output, args.output / name)
        print(f'[{index}/{len(jobs)}] {name}: {raw.shape[-1]} x {raw.shape[-2]}', flush=True)
    (args.output / 'run.json').write_text(json.dumps({
        'checkpoint': str(args.checkpoint.resolve()), 'mode': args.mode, 'device': args.device,
        'angle_order': ANGLES, 'samples': len(jobs), 'input_scale': args.input_scale,
        'output_units': {'s0': 'linear [0,1]', 'dolp': '[0,1]', 'aolp_rad': '[0,pi)',
                         'aolp_16bit.png': 'value / 65535 * pi radians'}}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
