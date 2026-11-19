"""V3 restoration and YOLO26s depth for four measured analyzer angles."""
from pathlib import Path
import os
import sys
import time
import cv2
import numpy as np
import torch
from paper_metrics import evaluate

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT / 'models' / 'v3'))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'work' / 'ultralytics')
os.environ['YOLO_AUTOINSTALL'] = 'false'
from infer import load_model, restore
from ultralytics import YOLO
import ultralytics.utils
ultralytics.utils.AUTOINSTALL = False

class Pipeline:
    def __init__(self):
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.restorer, self.cfg = load_model(ROOT / 'models/v3/weights/poldehazeV3_final.pt', self.device)
        self.depth = YOLO(str(ROOT / 'models/yolo26s-depth.pt'), task='depth')

    @torch.inference_mode()
    def predict_depth(self,image):
        b = np.uint8(np.clip(image,0,1)*255)
        rgb = np.repeat(b[:,:,None],3,axis=2)
        result = self.depth.predict(source=rgb,imgsz=768,device=self.device,verbose=False)[0]
        if result.depth is None: raise RuntimeError('The depth model returned no depth map.')
        d = result.depth.data.detach().float().cpu().numpy().squeeze()
        return cv2.resize(d,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_LINEAR)

    @torch.inference_mode()
    def run(self, angles, reference=None, display_shape=None, preserve_aspect=False):
        start = time.perf_counter()
        height,width=angles.shape[:2] if display_shape is None else display_shape
        preview_width=max(1,round(width*256/height))
        preview_angles=cv2.resize(angles,(preview_width,256),interpolation=cv2.INTER_AREA).astype(np.float32)
        model_size=(preview_width,256) if preserve_aspect else (256,256)
        resized = cv2.resize(angles, model_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        observed = np.clip(resized.sum(axis=2)/2,0,1)
        raw = torch.from_numpy(np.ascontiguousarray(resized.transpose(2,0,1)))[None].to(self.device)
        prepared_at = time.perf_counter()
        enhanced = restore(self.restorer,self.cfg,raw)[0,0].float().clamp(0,1).cpu().numpy().copy()
        restored_at = time.perf_counter()
        before, after = self.predict_depth(observed), self.predict_depth(enhanced)
        depths_at = time.perf_counter()
        if not all(np.isfinite(a).all() for a in (observed,enhanced,before,after)):
            raise ValueError('The model produced non-finite values.')
        metrics_started_at = time.perf_counter()
        metrics=evaluate(observed,enhanced,before,after,*reference) if reference is not None else None
        finished_at = time.perf_counter()
        return dict(angles=preview_angles,source_shape=(height,width),raw=observed,enhanced=enhanced,before=before,after=after,
                    reference_depth=None if reference is None else reference[1],metrics=metrics,
                    timings_ms=dict(dehazing=(restored_at-prepared_at)*1000,
                                    depth=(depths_at-restored_at)*1000,
                                    metrics=(finished_at-metrics_started_at)*1000 if reference is not None else None),
                    latency_ms=(finished_at-start)*1000)
