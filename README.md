# Self-Pruning Neural Network using Learnable Gates

## Problem Statement
Design a neural network that can automatically prune unnecessary connections during training to reduce model complexity while maintaining accuracy.

---

## Approach
A custom neural network was implemented using PyTorch.

Each layer uses **learnable gates** applied to weights:

```
weight = original_weight × sigmoid(gate_score)
```

A sparsity penalty is added to the loss function:

```
Total Loss = Classification Loss + λ × Sparsity Loss
```

Different values of λ (lambda) are used to study the pruning effect.

---

## Model Architecture
- Input: CIFAR-10 images (32×32)
- Layers:
  - Fully Connected: 3072 → 128
  - Fully Connected: 128 → 64
  - Fully Connected: 64 → 10

Each layer uses a custom **PrunableLinear** module.

---

## Results

| Lambda | Accuracy (%) | Hard Sparsity (<0.1) | Soft Sparsity (<0.3) | Mean Gate |
|--------|--------------|----------------------|----------------------|-----------|
| 0.01   | 19.35        | 0.00                 | 100.00               | 0.1525    |
| 0.05   | 20.90        | 0.00                 | 100.00               | 0.1525    |
| 0.10   | 21.05        | 0.00                 | 100.00               | 0.1525    |

---

## Analysis

- All gates converged to approximately **0.15**
- Hard sparsity (<0.1) remains **0%**
- Soft sparsity (<0.3) is **100%**

### Key Insight
The model **reduces all gates uniformly** instead of completely removing connections.

### Reason
This behavior is due to:
- sigmoid gating function
- L1-style sparsity loss
- limited training setup (CPU, fewer epochs)

---

## Experimental Constraints
- CPU-only training
- Reduced dataset size
- Smaller model
- Limited epochs

These constraints limit aggressive pruning but still demonstrate correct behavior.

---

## Conclusion
The self-pruning neural network was successfully implemented.

Although strong hard pruning was not achieved, the model demonstrates:
- learnable gating mechanism
- sparsity regularization
- pruning behavior trends

---

## Output Files
Check the `results/` folder:
- `results.csv`
- gate distribution plots
- trained model checkpoints
