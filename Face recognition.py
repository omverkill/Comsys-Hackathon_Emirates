# Siamese Network for Face Matching with Distorted Images
# --------------------------------------------------------
# This PyTorch implementation uses a pre-trained ResNet-50 as the base model
# to extract feature embeddings for pairs of face images and determines whether
# they belong to the same identity or not. Contrastive loss is used for training.

import os
import random
from PIL import Image
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score, accuracy_score

from google.colab import drive

# Mount Google Drive
# -------------------
drive.mount('/content/drive')

# Unzip the dataset from Google Drive
# ------------------------------------
!unzip -q '/content/drive/MyDrive/Comys_Hackathon5' -d /content

# Define dataset paths
base_path = '/content/Comys_Hackathon5/Task_B'
train_folder = os.path.join(base_path, 'train')
val_folder = os.path.join(base_path, 'val')

# Image transformation: resize and convert to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Custom Dataset for Siamese Training
# -----------------------------------
class SiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [cls for cls in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, cls))]
        self.images = []
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for img in os.listdir(cls_path):
                if img.endswith('.jpg') and 'distortion' not in img:
                    self.images.append((cls, os.path.join(cls_path, img)))

    def __getitem__(self, index):
        cls1, img1_path = self.images[index]
        img1 = Image.open(img1_path).convert('RGB')

        # Create a positive or negative pair
        if random.random() < 0.5:
            # Positive pair (same identity)
            cls2 = cls1
            folder = os.path.join(self.root_dir, cls2)
            candidates = [f for f in os.listdir(folder) if f != os.path.basename(img1_path) and 'distortion' not in f]
            if not candidates:
                return self.__getitem__((index + 1) % len(self))
            img2_path = os.path.join(folder, random.choice(candidates))
            label = 1.0
        else:
            # Negative pair (different identity)
            cls2 = random.choice([c for c in self.classes if c != cls1])
            other_folder = os.path.join(self.root_dir, cls2)
            candidates = [f for f in os.listdir(other_folder) if 'distortion' not in f]
            if not candidates:
                return self.__getitem__((index + 1) % len(self))
            img2_path = os.path.join(other_folder, random.choice(candidates))
            label = 0.0

        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)

    def __len__(self):
        return len(self.images)

# Siamese Network Model using ResNet-50
# --------------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        base_model.fc = nn.Linear(base_model.fc.in_features, 128)  # 128-dim embedding
        self.embedding = base_model

    def forward_once(self, x):
        return self.embedding(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# Contrastive Loss Function
# --------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = nn.functional.pairwise_distance(out1, out2)
        loss = label * dist.pow(2) + (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return loss.mean()

# Training Function
# ------------------
def train(model, loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for img1, img2, label in tqdm(loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

# Evaluation Helpers
# -------------------
@torch.no_grad()
def embed_image(model, path):
    image = Image.open(path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    return model.forward_once(image)

def build_reference_embeddings(model, ref_folder):
    ref_embeddings = {}
    for person in os.listdir(ref_folder):
        person_dir = os.path.join(ref_folder, person)
        if not os.path.isdir(person_dir): continue
        for img in os.listdir(person_dir):
            if 'distortion' in img or not img.endswith('.jpg'):
                continue
            img_path = os.path.join(person_dir, img)
            emb = embed_image(model, img_path)
            ref_embeddings.setdefault(person, []).append(emb)
    return ref_embeddings

def match_face_fast(test_emb, ref_embeddings, threshold=0.7):
    best_match, min_dist = None, float('inf')
    for person, embs in ref_embeddings.items():
        for ref_emb in embs:
            dist = nn.functional.pairwise_distance(test_emb, ref_emb).item()
            if dist < min_dist:
                min_dist = dist
                best_match = person
    label = 1 if min_dist < threshold else 0
    return best_match, label, min_dist

def evaluate_on_distorted_set_fast(model, distorted_root, ref_root, threshold=0.8):
    print("Precomputing reference embeddings...")
    reference_embeddings = build_reference_embeddings(model, ref_root)

    y_true, y_pred = [], []
    distorted_files = []
    for person in os.listdir(distorted_root):
        distort_dir = os.path.join(distorted_root, person, 'distortion')
        if os.path.isdir(distort_dir):
            for img in os.listdir(distort_dir):
                if img.endswith('.jpg'):
                    distorted_files.append((os.path.join(distort_dir, img), person))

    print(f"Total distorted images: {len(distorted_files)}")
    print("Running evaluation...")
    for img_path, true_id in tqdm(distorted_files):
        test_emb = embed_image(model, img_path)
        pred_id, pred_label, dist = match_face_fast(test_emb, reference_embeddings, threshold)
        match = int(pred_id == true_id)
        y_true.append(1)
        y_pred.append(match)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("✅ Evaluation Complete:")
    print(f"Top-1 Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Execution Block
# ----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Training and Evaluation
train_dataset = SiameseDataset(train_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train(model, train_loader, optimizer, criterion, epochs=10)
evaluate_on_distorted_set_fast(model, train_folder, train_folder, threshold=0.8)
