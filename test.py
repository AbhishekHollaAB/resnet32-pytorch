from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from model import ResNet32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) 
])

model = torch.load('ResNet_Weight.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader  = DataLoader(test_dataset, batch_size=1000)

images, labels = next(iter(test_loader))
images = images[:6].to(device)
outputs = model(images)
preds = torch.argmax(outputs, dim=1)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
    plt.title(f"Predicted: {preds[i].item()}")
    plt.axis('off')

plt.tight_layout()
plt.show()