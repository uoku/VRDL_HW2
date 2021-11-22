import json
from logging import error
import cv2
from numpy import errstate
from torch.cuda import device
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from torchvision import transforms
from utils.general import (LOGGER, check_img_size, non_max_suppression, xyxy2xywh, scale_coords)
import torch
import os
import numpy as np

dir = os.listdir("../test")
dir.sort(key=lambda x: int(x[:-4]))
answer = []

weights = './nowbestmodel/yolov5m6.pt'
d = select_device(0)
model = DetectMultiBackend(weights, device=d, dnn=False)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size([640, 640], s=stride)  # check image size
model(torch.zeros(1, 3, *imgsz).to(d).type_as(next(model.model.parameters())))  # warmup

model.eval()
for img_name in dir:
    print(img_name)
    img = cv2.imread("../test/" + img_name)
    org_img_shape = img.shape

    from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
    img = letterbox(img, 640, stride=stride, auto=pt and not jit)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(d)
    img = img.float()
    img /= 255
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    with torch.no_grad():
        pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    for i, det in enumerate(pred):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], org_img_shape).round()
        for *xyxy, conf, cls in reversed(det):
            # save json
            predic = dict()
            gn = torch.tensor(org_img_shape)[[1, 0, 1, 0]]  # normalization gain whwh
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            hei, wid = org_img_shape[:2]
            ax, ay, aw, ah = xywh[0] * wid, xywh[1] * hei, xywh[2] * wid, xywh[3] * hei

            new_x = ax - aw / 2
            new_y = ay - ah / 2
            new_width = aw
            new_heigh = ah

            predic['image_id'] = int(img_name[:-4])
            predic["bbox"] = [
                new_x,
                new_y,
                new_width,
                new_heigh
            ]
            predic['score'] = float(conf.item())
            predic["category_id"] = int(cls.item())
            answer.append(predic)

with open('answer.json', 'w') as f:
    json.dump(answer, f)
