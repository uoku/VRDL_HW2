
# My VRDL HW2

Use YOLOv5m6 pretrained model to do object detection

model from https://github.com/ultralytics/yolov5

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## My environment
GUP: GTX3060

CUDA: 11.3
cudnn: 8.2.1

## Preprocess
prepare dataset, config.yaml

1.change path setting in split_data_noscale.py and run
```data split
python split_data_noscale.py
```
2.set the path in my.yaml

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --img 640 --batch 16 --epochs 3 --data data/my.yaml --weights yolov5m6.pt
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/file/d/1bKORMAP306sk5m_d4swgeMfaHC3bUoBv/view?usp=sharing) 


## Inference
to reproduce submission file

chamge path of test img dir and weights at line 12,16,26

```Inference
 python predit.py
```

## colab

- [My colab](https://colab.research.google.com/drive/1vVYgnGcdu5aO37yeromgRU38csBUziJG?usp=sharing) 



