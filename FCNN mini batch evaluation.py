import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import random
import os

# Define dataset path
dataset_path = r"C:\Users\DELL\PycharmProjects\Deep Learning\dataset"

# Define dataset transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.6, 0.3, 0.5), (0.6, 0.3, 0.5))
])

# Load dataset
dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# Define Neural Network Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(224 * 224 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.fc4 = nn.Linear(10, 5)
        self.fc5 = nn.Linear(5, 2)  # Binary classification output

    def forward(self, x):
        x = x.view(-1, 224 * 224 * 3)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # Final output
        return x

# Initialize model, loss function, and optimizers
model = Net()
criterion = nn.CrossEntropyLoss()

# Optimizers
optimizers = {
    "SGD": torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8),
    "Adam": torch.optim.Adam(model.parameters(), lr=0.0005),
    "RMSprop": torch.optim.RMSprop(model.parameters(), lr=0.0007)
}

# Ensure GPU usage if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to train model with random mini-batch evaluations
def train_model(optimizer_name, optimizer, num_epochs=5):
    print(f"\nTraining with {optimizer_name} optimizer\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Perform random mini-batch evaluation
            if random.random() < 0.2:  # 20% chance to evaluate a batch
                eval_batch, eval_labels = next(iter(val_loader))
                eval_batch, eval_labels = eval_batch.to(device), eval_labels.to(device)
                model.eval()
                with torch.no_grad():
                    eval_outputs = model(eval_batch)
                    _, eval_preds = torch.max(eval_outputs, 1)
                    batch_acc = (eval_preds == eval_labels).sum().item() / eval_labels.size(0)
                print(f"Epoch {epoch+1}, Step {i+1}, Random Mini-Batch Accuracy: {batch_acc:.4f}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}")

    return model

# Function to test model
def test_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

# Train and test using different optimizers
results = {}

for optimizer_name, optimizer in optimizers.items():
    model = train_model(optimizer_name, optimizer, num_epochs=5)
    accuracy = test_model(model)
    results[optimizer_name] = accuracy
    print(f"Test Accuracy with {optimizer_name}: {accuracy:.4f}")

# Display results
print("\nFinal Optimization Comparison:")
for optimizer_name, accuracy in results.items():
    print(f"{optimizer_name}: Test Accuracy = {accuracy:.4f}")
