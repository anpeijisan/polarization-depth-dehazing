"""Per-frame metrics with paired intensity and same-estimator depth reference."""
import cv2
import numpy as np

METRICS = ('PSNR', 'SSIM', 'NMAE', 'AbsRel', 'SqRel', 'RMSE')

def image_scores(image, ref):
    a = np.asarray(image, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    mse = np.mean((a-b)**2)
    psnr = -10*np.log10(max(float(mse), 1e-12))
    # Exactly the standard 11x11, sigma 1.5 Gaussian, valid interior.
    filt = lambda x: cv2.GaussianBlur(x, (11,11), 1.5)[5:-5,5:-5]
    ma, mb = filt(a), filt(b)
    va, vb = filt(a*a)-ma*ma, filt(b*b)-mb*mb
    cov = filt(a*b)-ma*mb
    ssim = ((2*ma*mb+0.01**2)*(2*cov+0.03**2) /
            ((ma*ma+mb*mb+0.01**2)*(va+vb+0.03**2))).mean()
    return float(psnr), float(ssim)

def depth_scores(pred, ref):
    x = np.asarray(pred,dtype=np.float64); y = np.asarray(ref,dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    positive = valid & (y>1e-6)
    if not np.any(valid) or not np.any(positive):
        return (float('nan'),)*4
    delta = x[valid]-y[valid]
    span = float(np.percentile(y[valid],95)-np.percentile(y[valid],5))
    nmae = float(np.mean(np.abs(delta))/span) if span>1e-12 else float('nan')
    d = x[positive]-y[positive]; yr=y[positive]
    return (nmae, float(np.mean(np.abs(d)/yr)), float(np.mean(d*d/yr)),
            float(np.sqrt(np.mean(delta*delta))))

def evaluate(before_s0, after_s0, before_depth, after_depth, ref_s0, ref_depth):
    return (image_scores(before_s0,ref_s0)+depth_scores(before_depth,ref_depth),
            image_scores(after_s0,ref_s0)+depth_scores(after_depth,ref_depth))
