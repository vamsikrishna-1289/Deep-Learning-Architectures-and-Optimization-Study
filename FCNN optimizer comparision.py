import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# Define the dataset path
dataset_path = r"C:\Users\DELL\PycharmProjects\Deep Learning\dataset"

# Define the transforms for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.6, 0.3, 0.5), (0.6, 0.3, 0.5))  # Normalization values
])

# Load the dataset
dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

# Create a data loader for the dataset
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(224 * 224 * 3, 256)  # First layer
        self.fc2 = nn.Linear(256, 128)  # Second layer
        self.fc3 = nn.Linear(128, 10)  # Third layer
        self.fc4 = nn.Linear(10, 5)  # Fourth layer
        self.fc5 = nn.Linear(5, 2)  # Output layer for binary classification

    def forward(self, x):
        x = x.view(-1, 224 * 224 * 3)  # Flatten the input image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # Output layer
        return x

# Initialize the model, loss function, and optimizers
model = Net()
criterion = nn.CrossEntropyLoss()

# Define optimizers
optimizers = {
    "SGD": torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8),
    "Adam": torch.optim.Adam(model.parameters(), lr=0.0005),
    "RMSprop": torch.optim.RMSprop(model.parameters(), lr=0.0007)
}

# Reset model weights function
def reset_weights(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

# Function to train the model
def train_model(optimizer_name, optimizer, num_epochs=5):
    print(f"\nTraining with {optimizer_name} optimizer\n")

    # Reset the model weights
    model.apply(reset_weights)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(data_loader):.4f}')

    return model

# Function to test the model
def test_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

# Ensure to run on available device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training and testing with different optimizers
results = {}

for optimizer_name, optimizer in optimizers.items():
    model = train_model(optimizer_name, optimizer, num_epochs=5)
    accuracy = test_model(model)
    results[optimizer_name] = accuracy
    print(f"Test Accuracy with {optimizer_name}: {accuracy:.4f}")

# Summary of results
print("\nOptimization Comparison:")
for optimizer_name, accuracy in results.items():
    print(f"{optimizer_name}: Test Accuracy = {accuracy:.4f}")