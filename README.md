# ResNet32 - Custom Residual Network in PyTorch

This repository contains a custom implementation of a **ResNet32-like architecture** using PyTorch. It is inspired by the original [ResNet paper (He et al., 2015)](https://arxiv.org/abs/1512.03385) and tested on the **MNIST dataset**.

---

## Architecture

This custom ResNet32 uses the following configuration of residual blocks:

```
[3, 4, 5, 3] blocks in the 4 main layers  
(Each block is a modified BasicBlock with 2 Conv layers)
```

Similar to the original ResNet-34/50 which uses [3, 4, 6, 3], this version increases depth in the middle for experimentation and learning purposes.

---

## Features

- Fully custom BasicBlock with 3 convolutional layers
- Supports skip connections with downsampling
- Trained on MNIST (1-channel) by adjusting input channels
- Modular and clean class-based design
- Easy to extend to CIFAR-10 or ImageNet

---

## Dataset

Trained on **MNIST** using `torchvision.datasets.MNIST`.  
To support grayscale input, the first conv layer is modified from 3 channels to 1.

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/AbhishekHollaAB/resnet32-pytorch.git
cd resnet32-pytorch
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```

---

## Results

| Epochs | Accuracy |
|--------|----------|
| 5      | ~98.92%   |

> Note: Accuracy may vary slightly depending on hardware and random seed.

---

## File Structure

```
resnet32/
├── model.py       # Model definition
├── train.py          # Training and evaluation
├── test.py          # Testing
├── requirements.txt  # Python dependencies
└── README.md         # Project overview
```

---

## License

This project is licensed under the **MIT License** — feel free to use, modify, and share with attribution.

---

## Author

Built by [Abhishek Holla A B]  
📧 Email: abhiholla2012@gmail.com  
🌐 GitHub: [@AbhishekHollaAB](https://github.com/your-username)

---

## Star This Repo
