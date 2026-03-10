# Day 10 — Validation + overfitting + early stopping (CNN on FashionMNIST)

## What I implemented
- Created a validation split from the training set:
  - Train: 54,000
  - Val: 6,000
  - Test: 10,000 (official test set)
- Trained a CNN with early stopping logic (patience=2) based on `val_loss`
- Saved and restored the best checkpoint (best `val_loss`)

## Key terms (quick)
- Validation set: data used to make training decisions (epochs/hyperparameters), not for final reporting
- Overfitting: train improves while val/test stops improving
- Early stopping: stop training when `val_loss` does not improve for N epochs
- Checkpoint: saving the best model weights during training

## Training log (summary)
Epoch 1  | train_loss=0.5566 | val_loss=0.3842 | val_acc=0.8660  
Epoch 2  | train_loss=0.3497 | val_loss=0.3272 | val_acc=0.8840  
Epoch 3  | train_loss=0.2981 | val_loss=0.2914 | val_acc=0.8955  
Epoch 4  | train_loss=0.2663 | val_loss=0.2708 | val_acc=0.9010  
Epoch 5  | train_loss=0.2455 | val_loss=0.2840 | val_acc=0.8978  
Epoch 6  | train_loss=0.2260 | val_loss=0.2433 | val_acc=0.9122  
Epoch 7  | train_loss=0.2111 | val_loss=0.2455 | val_acc=0.9092  
Epoch 8  | train_loss=0.1966 | val_loss=0.2294 | val_acc=0.9160  
Epoch 9  | train_loss=0.1804 | val_loss=0.2300 | val_acc=0.9137  
Epoch 10 | train_loss=0.1706 | val_loss=0.2243 | val_acc=0.9197  

Best val_loss=0.2243 | Test loss=0.2436 | Test acc=0.9150

## Takeaways
- Validation metrics improved overall; some small fluctuations in val_loss are normal.
- Early stopping did not trigger within 10 epochs because val_loss continued to improve.
- The model generalizes well (val_acc ~0.92 and test_acc ~0.915).

## Next steps
1) Add dropout or weight decay and compare val/test behavior.
2) Try simple data augmentation (random crop/flip) and re-train.
3) Track curves (train/val loss per epoch) to visualize overfitting.
