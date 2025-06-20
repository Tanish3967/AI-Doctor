import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from tifffile import imread

# Path to the dataset
DATASET_PATH = r"C:\Users\KIIT\.cache\kagglehub\datasets\kmader\siim-medical-images\versions\6"
CSV_PATH = os.path.join(DATASET_PATH, "overview.csv")
IMG_DIR = os.path.join(DATASET_PATH, "tiff_images")

# Custom Dataset
class SIIMDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['tiff_name'])
        image = imread(img_name)
        image = Image.fromarray(image).convert("RGB")
        label = 1 if str(self.data.iloc[idx]['Contrast']).upper() == "TRUE" else 0
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader
dataset = SIIMDataset(CSV_PATH, IMG_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model (using a pretrained ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Modify the final layer for our binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: with contrast and without contrast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/10] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')

# Save the model
model_save_path = 'ct_contrast_classifier.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
