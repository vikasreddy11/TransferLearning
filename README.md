# Transfer Learning - Image Classification

Image classification using CNN and Transfer Learning with PyTorch on cat vs dog

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![GPU](https://img.shields.io/badge/GPU-RTX3050-green)

## Files
| File | Description |
|------|-------------|
| `convolution.py` | CNN from scratch |
| `feature_extraction.py` | Feature Extraction |
| `fine_tunning.py` | Fine Tuning |

## Model Architecture

### CNN from Scratch
```
Input (3×224×224)
    ↓
Conv2d(3→32) + ReLU + MaxPool
    ↓
Conv2d(32→64) + ReLU + MaxPool
    ↓
Conv2d(64→128) + ReLU + MaxPool
    ↓
Linear(128×28×28 → 512) + ReLU + Dropout
    ↓
Linear(512 → 2)
```

### ResNet50 Transfer Learning
```
ResNet50 Backbone (frozen)
    ↓
Linear(2048→512) + ReLU + Dropout
    ↓
Linear(512→2)
```
## Results

| Model |  AVG Accuracy |
|-------|----------|
| CNN from Scratch | 69.52% |
| Feature Extraction | 98.778% |
| Fine Tuning | 99.195% |

| Model |  BEST Accuracy |
|-------|----------|
| CNN from Scratch | 80.97% |
| Feature Extraction | 99.16% |
| Fine Tuning | 99.51% |


## Training curves
- The graph shows accuracy increasing and loss decreasing over 10 epochs.
- Train and Val curves are close together which means no overfitting.

CNN training graph
<img width="1200" height="400" alt="conv_train" src="https://github.com/user-attachments/assets/3ac52536-0844-4c32-9309-1afe593ad891" />

Feature Extraction
<img width="1200" height="400" alt="feature_E_training" src="https://github.com/user-attachments/assets/1e567790-39df-48ab-a0f3-572a2cf1abd7" />


Fine Tunning
<img width="1200" height="400" alt="Fine_training" src="https://github.com/user-attachments/assets/28c3289e-6883-4578-8953-a244139bdc90" />


## Confusion matrix
- Diagonal shows correct predictions.
- Some confusion between cat and dog which is expected as they look similar.

CNN confusion matrix
<img width="1200" height="800" alt="convConfusion" src="https://github.com/user-attachments/assets/3628b8db-56eb-4276-9315-2ff79fa9ec45" />

Feature Extraction confusion matrix
<img width="1000" height="800" alt="featureConfusion" src="https://github.com/user-attachments/assets/16b578da-de80-4720-bd32-a5b027ad391c" />


Fine Tunning confusion matrix
<img width="1000" height="800" alt="fineConfusion" src="https://github.com/user-attachments/assets/9d94eaec-3862-4fc1-aee7-0361133708e5" />


## Predictions
- Green title = correct prediction
- Red title = wrong prediction

CNN 
<img width="1600" height="300" alt="cnnpredicted" src="https://github.com/user-attachments/assets/00016de1-15b3-4a25-95e8-f45f22c16459" />

Feature Extraction
<img width="1600" height="300" alt="predicted_feature" src="https://github.com/user-attachments/assets/1d227a06-030b-4bc9-a92a-f75425f89fbd" />



Fine Tunning 
<img width="1600" height="300" alt="predicted_fine" src="https://github.com/user-attachments/assets/8e546bdc-4062-42cb-865f-01bed3b52f54" />

## What I learned

### CNN from Scratch
- Built a convolution layers to extract features from the images
- MaxPooling reduces the size of the image after each layer
- Dropout prevents overfitting 

### Transfer Learning
- ResNet50 pretrained on ImageNet already knows edges, shapes, textures
- No need to train from scratch — saves time and data

### Feature Extraction
- I used resnet50 
- Forze all backbone layers(requires_grad=False)
- Only trained the new classifier head 
- Fast training  only small number of parameters update

### Fine Tuning
- Froze backbone first, trained head
- Then unfroze last block (layer4) for resnet50

## How to Run

### Install dependencies
```bash
pip install torch torchvision matplotlib seaborn scikit-learn
```

### Run CNN from scratch
```bash
python convolution.py
```

### Run Feature Extraction
```bash
python feature_extraction.py
```

### Run Fine Tuning
```bash
python fine_tunning.py
```

## Author
**Vikas Reddy**
- GitHub: [@vikasreddy11](https://github.com/vikasreddy11)
- LinkedIn: [Vikas Reddy](https://www.linkedin.com/in/vikas-reddy-veeramreddy-26057138a)
