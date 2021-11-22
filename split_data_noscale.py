import os
import h5py
from PIL import Image
import numpy as np
from torchvision import transforms
import random

data = os.listdir('train')

total = len(data)

source = 'train/'

target = 'dataset_noscale/images/train/'
target_val = 'dataset_noscale/images/val/'

f = h5py.File('digitStruct.mat', 'r')

for idx in range(total):
    val = False
    if random.random() < 0.05:
        val = True

    img_path = os.path.join('train/', str(idx+1)+".png")
    img = Image.open(img_path).convert('RGB')

    print(img_path)
   
    bbox_prop = ['height', 'left', 'top', 'width', 'label']
        
    def get_boxes(f, idx=0):
        meta = {key:[] for key in bbox_prop}
        bbox = f['digitStruct/bbox']
        box = f[bbox[idx][0]]
        for key in box.keys():
            if box[key].shape[0] == 1:
                meta[key].append(int(box[key][0][0]))
            else:
                for i in range(box[key].shape[0]):
                    meta[key].append(int(f[box[key][i][0]][()].item()))
        return meta
    
    # resize image and save
    w, h = img.size
    if val == True:
        img.save(target_val+str(idx+1)+".png")
    else:
        img.save(target+str(idx+1)+".png")

    # cal target and save
    if val == True:
        f = open('dataset_noscale/labels/val/'+str(idx+1)+".txt", 'w')
    else:
        f = open('dataset_noscale/labels/train/'+str(idx+1)+".txt", 'w')
    h5 = h5py.File('digitStruct.mat', 'r')
    detail = get_boxes(h5, idx)
    for i in range(len(detail['label'])):
        label = str(detail['label'][i])
        if label == '10':
            label = '0'
        height = np.float64(detail['height'][i])
        left = np.float64(detail['left'][i])
        top = np.float64(detail['top'][i])
        width = np.float64(detail['width'][i])
        center_x = left + width/2
        center_y = top + height/2

        # normalize
        new_x = center_x / w
        new_y = center_y / h
        new_width = width / w
        new_height = height / h

        f.write(label+' '+str(new_x)+' '+str(new_y)+' '+str(new_width)+' '+str(new_height)+'\n')
    f.close()

print('OK')



