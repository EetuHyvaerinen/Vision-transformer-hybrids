# Vision Transformer Hybrid Architectures

Implementation and experimentation with Vision Transformers (ViT), DeiT-style knowledge distillation, and CNN–Transformer hybrid architectures using PyTorch.

This project explores how inductive biases from convolutional neural networks can improve transformer-based vision models, particularly on small datasets.

Experiments were conducted on the FashionMNIST dataset.

---

# Project Goals

Vision Transformers achieve state-of-the-art results on large datasets but struggle with smaller datasets due to a lack of inductive bias.

This project investigates three solutions:

1. Knowledge Distillation from CNN → Vision Transformer
2. DeiT-style transformer architecture with distillation token
3. Hybrid CNN–Transformer architectures

---

# Architectures Implemented

## 1. CNN Teacher Model

A convolutional neural network trained to act as a teacher model for knowledge distillation.

Architecture:

Conv → ReLU → MaxPool
Conv → ReLU → MaxPool
Fully Connected → Dropout → Classifier

**Test Accuracy:** 91.05%

---

# 2. Vision Transformer (Baseline)

A Vision Transformer implemented from scratch, including:

* patch embedding
* multi-head self-attention
* transformer encoder blocks
* classification token

Baseline result:

**Test Accuracy:** ~79%

---

# 3. Knowledge Distillation

A Hard Distillation Loss was implemented to train the ViT using the CNN teacher.

Loss function:

L = 0.5 * CE(student, labels) + 0.5 * CE(student, teacher_labels)

Teacher predictions are used as additional supervision.

**Distilled ViT Accuracy:** 82.7%

---

# 4. Custom DeiT Architecture

The Vision Transformer was extended with a distillation token similar to the DeiT architecture proposed by Meta AI.

Architecture changes:

* Distillation token added to the embedding layer
* Modified positional embeddings
* Dual-head classifier for class and distillation tokens

**MyDeiT Accuracy:** 82.99%

---

# 5. Hybrid CNN–Transformer Architectures

Two hybrid approaches were implemented.

## Serial Integration

CNN used as a feature extractor feeding feature maps into the Vision Transformer.

Image → CNN → ViT → Classifier

**Test Accuracy:** 88.19%

---

## Late Fusion

Two feature extractors operate independently:

CNN Feature Extractor
Vision Transformer Feature Extractor

Their outputs are concatenated and passed through a fusion classifier.

**Test Accuracy:** 90.03%

---

# Experimental Results

| Model                 | Test Accuracy |
| --------------------- | ------------- |
| CNN Teacher           | 91.05%        |
| ViT Baseline          | ~79%          |
| Distilled ViT         | 82.7%         |
| MyDeiT                | 82.99%        |
| Hybrid CNN → ViT      | 88.19%        |
| Late Fusion CNN + ViT | 90.03%        |

---

# Key Insights

CNN inductive biases help Vision Transformers learn better representations on small datasets.

Knowledge distillation improves transformer training efficiency.

Hybrid CNN–Transformer architectures combine the strengths of both approaches and outperform standalone ViT models.

---

# Technologies Used

* PyTorch
* Vision Transformers
* Knowledge Distillation
* CNN architectures

---

# Future Work

Possible improvements:

* larger datasets (CIFAR-10 / ImageNet subsets)
* soft distillation loss
* attention visualization
* larger transformer architectures
