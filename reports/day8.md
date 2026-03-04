# Day 8 — PyTorch basics (MLP on FashionMNIST)

## What I built
- Implemented the core PyTorch training loop: forward → loss → backward → optimizer.step()
- Trained a simple MLP for image classification on FashionMNIST (CPU)

## Data
- Dataset: FashionMNIST
- Input shape: [batch, 1, 28, 28]
- Batch size: 128 (train), 256 (test)
- Device: CPU

## Model
- Architecture: Flatten → Linear(784→256) → ReLU → Linear(256→10)
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=1e-3)

## Results (2 epochs)
- Epoch 1: loss=0.5618, train_acc=0.8394, test_acc=0.8252
- Epoch 2: loss=0.4016, train_acc=0.8658, test_acc=0.8520

## Notes / Takeaways
- Tensors are the core data structure; autograd computes gradients automatically.
- Training updates model weights; inference does not.
- Next: add validation split + try a CNN (conv layers) to improve accuracy.
