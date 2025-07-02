# --- 1. Setup and Environment ---

# Import necessary libraries
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
# matplotlib.pyplot is imported but not used in the provided snippet.
# from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from google.colab import drive

# Mount Google Drive to access the dataset
print("Mounting Google Drive...")
drive.mount('/content/drive')
print("Google Drive mounted successfully.")

# Unzip dataset into a specified directory
# The -q flag is for quiet mode (no verbose output during unzipping)
# The -d flag specifies the destination directory
print("Unzipping dataset...")
!unzip -q '/content/drive/MyDrive/Cosmys/Comys_Hackathon5 new.zip' -d /content/gender_data
print("Dataset unzipped to /content/gender_data.")

# --- 2. Data Preparation ---

# Define image transformations for training and validation phases
# These transformations preprocess images for the neural network
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),       # Resize images to 224x224 pixels (standard for MobileNetV2)
        transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally for data augmentation
        transforms.ToTensor(),               # Convert PIL Image to PyTorch Tensor (scales to [0, 1])
        transforms.Normalize([0.5]*3, [0.5]*3) # Normalize pixel values to [-1, 1] for each RGB channel
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),       # Resize validation images to 224x224
        transforms.ToTensor(),               # Convert to Tensor
        transforms.Normalize([0.5]*3, [0.5]*3) # Normalize pixel values
    ]),
}

# Define the root directory where the dataset is located after unzipping
data_dir = '/content/gender_data/Comys_Hackathon5/Task_A'

# Create PyTorch datasets using ImageFolder, which expects images
# organized in subdirectories where each subdirectory name is a class label.
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}

# Create DataLoaders for iterating over datasets in batches
# shuffle=True for training data ensures better generalization
# num_workers=2 enables multi-process data loading for speed
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=2)
               for x in ['train', 'val']}

# Get the class names (e.g., 'male', 'female') from the training dataset
class_names = image_datasets['train'].classes
print(f"Detected classes: {class_names}")

# --- 3. Model Architecture ---

# Determine the device to run the model on (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a pre-trained MobileNetV2 model
# pretrained=True loads weights trained on the ImageNet dataset
model = models.mobilenet_v2(weights=models.MobileNetV2_Weights.IMAGENET1K_V1)

# Freeze all parameters in the feature extractor part of the model
# This is a common practice in transfer learning to leverage learned features
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier head of MobileNetV2 for binary classification
# MobileNetV2's classifier is a Sequential module, with the final linear layer at index 1
# model.last_channel gives the input features to the original classifier
# We replace it with a new Linear layer that outputs 2 classes (male/female)
model.classifier[1] = nn.Linear(model.last_channel, 2)

# Move the model to the selected device (GPU or CPU)
model = model.to(device)
print("MobileNetV2 model loaded and adapted for 2 classes.")

# --- 4. Loss Function and Optimizer ---

# Define the loss function: Cross-Entropy Loss for classification tasks
criterion = nn.CrossEntropyLoss()

# Define the optimizer: Adam optimizer
# Only the parameters of the new classifier layer are passed to the optimizer,
# as the rest of the model (frozen backbone) does not require gradient updates.
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
print("Criterion (CrossEntropyLoss) and Optimizer (Adam) initialized.")

# --- 5. Training Function ---

def train_model(model, criterion, optimizer, num_epochs=5):
    """
    Trains and validates the given model over multiple epochs.

    Args:
        model (torch.nn.Module): The neural network model to train.
        criterion (torch.nn.modules.loss._Loss): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for updating weights.
        num_epochs (int): The number of training epochs.

    Returns:
        torch.nn.Module: The trained model.
    """
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode (e.g., enables dropout)
            else:
                model.eval()   # Set model to evaluation mode (e.g., disables dropout/batchnorm updates)

            running_loss = 0.0
            running_corrects = 0
            all_preds = []     # List to store all predictions for F1, Precision, Recall
            all_labels = []    # List to store all true labels for F1, Precision, Recall

            # Iterate over data batches
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device) # Move data to the configured device

                optimizer.zero_grad() # Zero the parameter gradients before each batch

                # Forward pass: enable gradients only in the training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)            # Get model predictions (logits)
                    _, preds = torch.max(outputs, 1)   # Get the predicted class index (0 or 1)
                    loss = criterion(outputs, labels)  # Calculate batch loss

                    # Backward + optimize only if in the training phase
                    if phase == 'train':
                        loss.backward()  # Compute gradients of the loss with respect to model parameters
                        optimizer.step() # Update model weights using the optimizer

                running_loss += loss.item() * inputs.size(0) # Accumulate weighted loss for the epoch
                running_corrects += torch.sum(preds == labels.data) # Accumulate correct predictions

                # Store predictions and labels for comprehensive metrics calculation later
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Calculate and print epoch-level metrics
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            # Calculate and print additional metrics (Precision, Recall, F1-Score) for the validation phase
            if phase == 'val':
                # 'binary' average is used because we have two classes and it's a binary classification task.
                # If your labels are not 0 and 1, you might need to specify pos_label if the 'positive' class
                # isn't implicitly handled by sklearn as label 1.
                precision = precision_score(all_labels, all_preds, average='binary')
                recall = recall_score(all_labels, all_preds, average='binary')
                f1 = f1_score(all_labels, all_preds, average='binary')
                print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    return model

# --- 6. Model Training and Saving ---

print("\nStarting model training...")
# Call the train_model function to begin the training process
trained_model = train_model(model, criterion, optimizer, num_epochs=5)
print("\nModel training complete.")

# Save the state dictionary of the trained model
# This saves only the learned weights, not the entire model architecture.
# It allows for loading the trained model later without retraining.
model_save_path = 'gender_classifier.pth'
torch.save(trained_model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")

# --- End of Script ---