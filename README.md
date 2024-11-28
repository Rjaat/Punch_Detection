# Boxing Punch Detection using YOLOv7
This README file contains instructions for the implementation of boxing punch detection system using YOLOv7. The model can detect and classify four types of punches: jab, cross, hook, and uppercut.

## Model Performance
- **mAP@0.5**: 0.993
- **Individual Class Performance**:
  - Jab: 0.995 mAP
  - Cross: 0.986 mAP
  - Hook: 0.995 mAP
  - Uppercut: 0.996 mAP

## Installation

### 1. Downloading the Project Code and Related files i.e [Dataset](https://drive.google.com/drive/folders/1qzPn99GcrFnrzayfp5X6pS46BK4dRpgK?usp=drive_link)

[[Google-Drive Link](https://drive.google.com/drive/folders/1HRDerp0pccm8AIHUzREqMQJd99THlhmR?usp=drive_link)]


### 2. Create and Activate Conda Environment
```bash
# Create conda environment
conda create -n boxing-yolo python=3.8
conda activate boxing-yolo

# Install requirements
pip install -r requirements.txt
```

### 3. Download Weights
The latest trained weights are available in the `boxing_detection3` folder. You can also download them from:
[[Google Drive Link](https://drive.google.com/drive/folders/1mKPCqtykC1ovuwiOTAXEzXD15M1XNIal?usp=drive_link)]



## Usage

### Testing
To test the model using the latest weights:
```bash
python3 test.py --data yolo_dataset/dataset.yaml \
                --weights weights/best.pt \
                --batch-size 16 \
                --img-size 640 
               
```


### Training (if needed)
```bash
python3 train.py --data yolo_dataset/dataset.yaml \
                 --cfg cfg/training/yolov7.yaml \
                 --weights weights/yolov7.pt \
                 --batch-size 16 \
                 --epochs 300 \
                 --name boxing_detection 
                 
```

## Dataset Format
The dataset follows the YOLOv7 format:
- Images: `.png` format in `yolo_dataset/images/`
- Labels: `.txt` format in `yolo_dataset/labels/`
- Classes: 
  - 0: jab
  - 1: cross
  - 2: hook
  - 3: uppercut

## Results
The model achieves robust performance across different punch types:
```
Class        Precision    Recall    mAP@0.5
--------------------------------------------
jab          1.000       0.990     0.995
cross        0.920       0.980     0.986
hook         1.000       0.970     0.995
uppercut     1.000       0.980     0.996
--------------------------------------------
Average      0.980       0.980     0.993
```

## Common Issues and Solutions

### CUDA Out of Memory
If you encounter CUDA out of memory errors:
```bash
# Reduce batch size
python3 test.py --batch-size 4 --data yolo_dataset/dataset.yaml
```

### CPU-Only Testing
For systems without GPU:
```bash
python3 test.py --data yolo_dataset/dataset.yaml --device cpu
```


## Acknowledgments
- YOLOv7 Implementation: https://github.com/WongKinYiu/yolov7
- Dataset creation and annotation tools: CVAT

