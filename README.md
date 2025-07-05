# Comsys-Hackathon_Emirates
#  Face Recognition & Gender Classification using Deep Learning

This repository contains two independent deep learning models:

- 👩‍🦰 Gender Classification using MobileNetV2
- 🧑‍🦱 Face Matching with Distorted Inputs using Siamese Network (ResNet-50)

Both models are implemented using PyTorch and designed for robust performance in real-world face analysis tasks.

---

##  Directory Structure

```
.
├── Face recognition.py         # Siamese network for distorted face matching
├── GenderClassification.py     # Gender classifier using MobileNetV2
├── genderModel.ipynb           # Notebook for gender classification training
├── siamese_resnet50_7epochs.ipynb # (Optional) Notebook variant for face matching
├── README.md                   # Project documentation
```

---

## 🔧 Requirements

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- PIL (Pillow)
- tqdm
- Google Colab (or local GPU environment)

---

##  Task A: Gender Classification

###  Dataset Format

The gender dataset should follow this folder structure:

```
Task_A/
├── train/
│   ├── male/
│   └── female/
└── val/
    ├── male/
    └── female/
```

###  Model

- Pretrained MobileNetV2 used as feature extractor.
- Final classifier modified for binary classification (male vs. female).

###  Metrics

- Accuracy
- Precision
- Recall
- F1-Score

### 🏁 Training

Run:  
```bash
python GenderClassification.py
```

### Output

- Model is saved to: gender_classifier.pth

---

##  Task B: Face Matching (Distorted Face Verification)

###  Dataset Format

Structure:

```
Task_B/
├── train/
│   ├── person1/
│   │   ├── image1.jpg
│   │   └── distortion/
│   │       └── distorted1.jpg
│   ├── person2/
│   └── ...
├── val/ (same structure as train/)
```

###  Model

- Siamese Network using ResNet-50 as feature extractor.
- Learns to embed face pairs and distinguish identities using contrastive loss.

###  Evaluation

- Compares distorted images to clean reference images.
- Reports Top-1 Accuracy and F1-Score.

###  Training

Run:  
```bash
python Face\ recognition.py
```

> Default: 7 epochs with batch size 16.

---

##  Pretrained Weights

After training, model weights can be saved and reused:

- `gender_classifier.pth`: MobileNetV2 gender classifier
- `siamese_resnet50.pth`: Siamese network for face matching (optional: save via torch.save)

---

##  Performance Metrics

| Task              | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Gender Classification | ~92% | ~87%     | ~98%  | ~93%    |
| Face Matching         | ~% |    -      |   -    | ~%    |

---

## Highlights

-  Works with distorted/augmented test faces.
-  Supports transfer learning & metric learning.
-  Fast embedding and inference with pretrained backbones.

---

##  Acknowledgments

- PyTorch and torchvision teams
- ResNet & MobileNet research communities
- Cosmys Hackathon 5 Challenge

