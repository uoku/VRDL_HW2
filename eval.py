import torch
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import cv2
import numpy as np
from pathlib import Path
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox

@torch.no_grad()
def pred():
    weights = 'runs\\train\exp7\weights\\best.pt'
    imgsz = 480

    save_dir = increment_path(Path('runs/detect') / 'exp', exist_ok=False)  # increment run
    (save_dir / 'labels' if False else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    device = select_device(0)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.model.float()

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup

    path ='../dataset/images/train/1.png'
    img = cv2.imread(path)
    img = letterbox(img, imgsz, stride=stride, auto=pt)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()

    img /= 255
    visualize = increment_path(save_dir / Path(path).stem, mkdir=True)
    pred = model(img, augment=False, visualize=False)

    print(pred)

if __name__ == '__main__':
    pred()