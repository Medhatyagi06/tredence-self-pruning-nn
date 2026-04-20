# Self-Pruning Neural Network

## Problem
Design a neural network that prunes itself during training.

## Method
Each weight has a gate value:
weight = original_weight * sigmoid(gate_score)

Loss = Classification Loss + lambda * sum(gates)

## Why L1 works
L1 pushes values to zero → gates go to zero → weights removed.

## Results

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| Fill after running |

## Output
Check results folder:
- results.csv
- gate plots