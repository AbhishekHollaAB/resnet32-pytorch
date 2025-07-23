from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ResNet32, BasicBlock
import torch
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # mean and std for MNIST
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet32().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

nEpochs = 5
for epoch in tqdm(range(nEpochs)):
    model.train()
    totalLoss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        totalLoss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {totalLoss:.4f}')

torch.save(model, 'ResNet_Weight.pth')
print('Training Complete! Now Evaluating the Performance on the test set...')
print()
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')