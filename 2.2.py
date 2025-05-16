import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from kagglehub import dataset_download
from pathlib import Path

root_path = dataset_download("vishesh1412/celebrity-face-image-dataset")
DATA_DIR = Path(root_path) / "Celebrity Faces Dataset"
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NUM_EPOCHS = 15 # 10
LEARNING_RATE = 0.001
MODEL_PATH = "modelo_identificacion_personas.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== FUNCIONES ==========
def create_dataloaders(data_dir, image_size, batch_size, split_ratio=0.8):
    transform_train = transforms.Compose([
        transforms.Resize(image_size),  # Pass image_size directly
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),  # Pass image_size directly
        transforms.ToTensor(),
    ])

    # Carga el dataset sin transform porque lo aplicaremos por separado
    base_dataset = datasets.ImageFolder(data_dir)
    class_names = base_dataset.classes

    total_size = len(base_dataset)
    train_size = int(split_ratio * total_size)
    test_size = total_size - train_size
    generator = torch.Generator().manual_seed(42)
    train_indices, test_indices = torch.utils.data.random_split(
        list(range(total_size)), [train_size, test_size], generator=generator
    )

    # Aplica transformaciones específicas a cada subconjunto
    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(data_dir, transform=transform_train), train_indices
    )
    test_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(data_dir, transform=transform_test), test_indices
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, class_names
# no 25 ni 10
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=15):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader, class_names, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def build_model(num_classes):
    model = models.resnet18(weights='IMAGENET1K_V1')


    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

# ========== PROCESO PRINCIPAL ==========
def main():
    print("Cargando datos...")
    train_loader, test_loader, class_names = create_dataloaders(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    print(f"Clases detectadas: {class_names}")

    model = build_model(num_classes=len(class_names)) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    print("Entrenando modelo...")
    train_model(model, train_loader, criterion, optimizer, DEVICE, NUM_EPOCHS)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

    print("Evaluando modelo...")
    evaluate_model(model, test_loader, class_names, DEVICE)

if __name__ == "__main__":
    main()