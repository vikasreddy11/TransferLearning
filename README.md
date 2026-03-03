# Transfer Learning - Image Classification

Image classification using CNN and Transfer Learning with PyTorch on cat vs dog

## Files
| File | Description |
|------|-------------|
| `convolution.py` | CNN from scratch |
| `feature_extraction.py` | Feature Extraction |
| `fine_tunning.py` | Fine Tuning |


## Results

| Model |  AVG Accuracy |
|-------|----------|
| CNN from Scratch | 69.52% |
| Feature Extraction | 98.61% |
| Fine Tuning | xx% |

| Model |  BEST Accuracy |
|-------|----------|
| CNN from Scratch | 80.97% |
| Feature Extraction | 99.21% |
| Fine Tuning | xx% |


## Training curves
-The graph shows accuracy increasing and loss decreasing over 10 epochs.
-Train and Val curves are close together which means no overfitting.

CNN training graph
<img width="1200" height="400" alt="conv_train" src="https://github.com/user-attachments/assets/3ac52536-0844-4c32-9309-1afe593ad891" />

Feature Extraction
<img width="1200" height="400" alt="feature_training" src="https://github.com/user-attachments/assets/f0a123e7-8d56-42b2-bd86-98f027496cfd" />

Fine Tunning
<img width="1200" height="400" alt="Fine_training" src="https://github.com/user-attachments/assets/2c96e738-1cca-4cc0-9235-4494283e888c" />

## Confusion matrix
-Diagonal shows correct predictions.
-Some confusion between cat and dog which is expected as they look similar.

CNN confusion matrix
<img width="1200" height="800" alt="convConfusion" src="https://github.com/user-attachments/assets/3628b8db-56eb-4276-9315-2ff79fa9ec45" />

Feature Extraction confusion matrix
<img width="1000" height="800" alt="featureConfusion" src="https://github.com/user-attachments/assets/3d6a81df-97b9-4308-8126-d48f2e3cf81b" />

Fine Tunning confusion matrix
<img width="1000" height="800" alt="fineConfusion" src="https://github.com/user-attachments/assets/74148d40-7648-464c-9a14-cdee3e17ceaf" />

## Predictions
-Green title = correct prediction
-Red title = wrong prediction

CNN 
<img width="1600" height="300" alt="cnnpredicted" src="https://github.com/user-attachments/assets/00016de1-15b3-4a25-95e8-f45f22c16459" />

Feature Extraction
<img width="1600" height="300" alt="predicted_feature" src="https://github.com/user-attachments/assets/1fc28614-bb95-4226-a019-93d544025b49" />

Fine Tunning 
<img width="1600" height="300" alt="predicted_fine" src="https://github.com/user-attachments/assets/cef2fe8e-993c-43b0-b982-71fb922d7cde" />

## What I learned

### CNN from Scratch
-Built a convolution layers to extract features from the images
-MaxPooling reduces the size of the image after each layer
-Dropout prevents overfitting 

### Transfer Learning
- ResNet50 pretrained on ImageNet already knows edges, shapes, textures
- No need to train from scratch — saves time and data

### Feature Extraction
-I used resnet50 
-Forze all backbone layers(requires_grad=False)
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
- LinkedIn: [@linkedin.com/in/vikas-reddy-veeramreddy-26057138a]
