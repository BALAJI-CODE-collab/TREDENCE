# PruneVision Report

## Introduction

PruneVision combines a defensive preprocessing pipeline with a prunable neural network to detect plant disease while exposing the trade-off between accuracy and sparsity. The goal is not only to classify images, but also to reject poor inputs, explain confidence, and reduce unnecessary parameters through learned gates.

## Why L1 Penalty on Sigmoid Gates Encourages Sparsity

Each prunable linear layer uses gates computed as:

$$g = \sigma(s)$$

where $s$ is the learned gate score tensor and $\sigma$ is the sigmoid function. The effective weight becomes:

$$W_{eff} = W \odot g$$

The training objective is:

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_{sparse} \|g\|_1$$


Because the gates live in $[0, 1]$, the L1 term directly penalizes large active gate values. During optimization, the model can lower the loss by pushing unnecessary gates toward zero, which reduces the contribution of those weights. Intuitively, the network learns to turn off neurons that do not help classification, leaving a smaller active subnetwork.

The larger the $\lambda_{sparse}$ value, the stronger the pressure to minimize active gates. That improves sparsity, but it can also suppress useful parameters and reduce accuracy if the penalty is too strong.

## Results

Populate this table from `results.json` after running `backend/train.py`.
Before case-study submission, replace all "Pending training" cells with the actual values produced by `backend/train.py`.

| Lambda | Test Accuracy | Sparsity Level % |
|---|---:|---:|
| 0.0001 | Pending training | Pending training |
| 0.001 | Pending training | Pending training |
| 0.01 | Pending training | Pending training |

## Lambda Trade-Off Analysis

Small values of $\lambda_{sparse}$ usually preserve accuracy because the model is only lightly encouraged to prune. Mid-range values often give the best balance between compactness and performance. Very large values can over-prune, reducing representational capacity and causing accuracy to fall.

In practice, the best lambda is the one that achieves enough sparsity to reduce size and improve interpretability without materially degrading test accuracy.

## Intelligent Preprocessor Impact on Accuracy

The preprocessing pipeline improves robustness by rejecting low-quality images, reducing noise, normalizing contrast, and isolating the leaf region. This reduces the amount of background clutter the model sees and makes the input distribution more consistent between training and inference. The quality gate also provides a meaningful failure path when the image is too blurry, too dark, too bright, or too small.

## Gate Distribution Plot Description

The gate distribution plot shows how many learned gates remain near 1.0, how many move toward 0.0, and how aggressively the pruning penalty shaped the final model. A distribution concentrated near the extremes indicates stronger specialization. A wide concentration near 0.5 indicates a softer, less decisive pruning pattern.

## Conclusion

PruneVision demonstrates how preprocessing, explainable confidence gating, and learnable sparsity can work together in an applied AI system. The framework is practical for deployment because it exposes quality diagnostics, supports model statistics at runtime, and keeps the inference stack small enough to evolve from a prototype to a production service.
