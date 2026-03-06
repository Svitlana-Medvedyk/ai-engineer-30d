# Day 9 — CNN vs MLP on FashionMNIST

## Goal
Replace MLP with a CNN and compare performance fairly (same dataset, 2 epochs).

## Model (CNN)
- Conv2d(1→32, k=3, pad=1) + ReLU + MaxPool2d(2)
- Conv2d(32→64, k=3, pad=1) + ReLU + MaxPool2d(2)
- Flatten → Linear(64*7*7→128) + ReLU → Linear(128→10)
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=1e-3)
- Device: CPU

## Results
- CNN Epoch 1: loss=0.5003, train_acc=0.8788, test_acc=0.8689
- CNN Epoch 2: loss=0.3138, train_acc=0.8989, test_acc=0.8861

## Comparison
- MLP test_acc (Day 8): 0.8520
- CNN test_acc (Day 9): 0.8861
- Delta: +0.0341

## Notes / Takeaways
- CNN outperformed MLP because convolutions capture local spatial patterns (edges/textures) and pooling adds robustness.
- Next: add validation split + try dropout or train 5 epochs and monitor overfitting (train vs test gap).
