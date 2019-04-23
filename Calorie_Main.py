import os
import torch 
import torch.nn as nn
#from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_data_path = r'C:\Users\user\Desktop\Dataset\Train data'
test_data_path = r'C:\Users\user\Desktop\Dataset\Test data'
# Hyper parameters
num_epochs = 3
num_classes = 12
batch_size = 32
learning_rate = 0.001

def load_training(train_data_path, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([64,64]),
         transforms.RandomCrop(64),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=train_data_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader

def load_test(test_data_path, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([64,64]),
         transforms.RandomCrop(64),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return test_loader



# Data loader
train_load = load_training(train_data_path, batch_size)

test_load = load_test(test_data_path, batch_size)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(8192, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

# Train the model
total_step = len(train_load)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_load):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_load:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
