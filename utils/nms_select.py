
from .config import cfg
from .nms_cpu import nms
from .gpu_nms import gpu_nms

def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return nms(dets, thresh)
