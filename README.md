# NLP Advanced Project: Deceptive Opinion Spam Detection

## Scientific Abstract

This research project implements a comprehensive natural language processing pipeline for the detection and classification of deceptive opinion spam in hotel reviews. The system combines classical text feature engineering with state-of-the-art deep learning architectures, specifically Bidirectional Long Short-Term Memory (Bi-LSTM) networks and fine-tuned transformer models (DistilBERT), to achieve robust discrimination between authentic and fraudulent reviews.

---

## Table of Contents
1. [Research Objectives](#research-objectives)
2. [Dataset Specification](#dataset-specification)
3. [System Architecture](#system-architecture)
4. [Methodology](#methodology)
5. [Technical Implementation](#technical-implementation)
6. [Performance Analysis](#performance-analysis)
7. [Installation & Execution](#installation--execution)

---

## Research Objectives

### Primary Goals

1. **Exploratory Data Analysis (EDA)**: Characterize linguistic and statistical features distinguishing truthful from deceptive reviews
2. **Deep Learning Implementation**: Develop and optimize a Bi-LSTM neural architecture in PyTorch for sequence classification
3. **Cross-Validation Strategy**: Implement K-Fold cross-validation to mitigate overfitting on limited dataset (n=1600)
4. **Transformer Baseline**: Fine-tune DistilBERT for comparative evaluation against recurrent architectures
5. **Regularization Analysis**: Comprehensive visualization and interpretation of training dynamics
6. **Interactive Deployment**: Create real-time inference interface using Gradio

---

## Dataset Specification

### Data Characteristics

**Dataset**: Deceptive Opinion Spam Corpus (from Kaggle)
- **Total Samples**: 1,600 reviews
- **Balance**: 800 truthful / 800 deceptive
- **Source**: TripAdvisor, Mechanical Turk
- **Domain**: Hotel industry

### Feature Schema

| Feature | Type | Description |
|---------|------|-------------|
| `deceptive` | Categorical | Binary classification: truthful/deceptive |
| `hotel` | Categorical | Hotel chain identifier (20 unique properties) |
| `polarity` | Categorical | Review sentiment: positive/negative |
| `source` | Categorical | Review platform origin |
| `text` | String | Raw review content |
| `label` | Integer | Binary encoding (0: truthful, 1: deceptive) |

### Statistical Properties

**Text Feature Comparison**:
| Metric | Truthful | Deceptive | Δ |
|--------|----------|-----------|-------|
| Mean Length | 821.0 chars | 791.8 chars | -3.6% |
| Mean Word Count | 171.5 | 163.4 | -4.7% |
| Mean Punctuation | 23.87 | 18.02 | -24.5% ← **Significant** |
| Mean Stopword Count | 74.95 | 75.80 | +1.1% |

**Key Finding**: Deceptive reviews exhibit 24.5% fewer punctuation marks, suggesting stylistic compression and reduced emotional expression.

---

## System Architecture

### 1. Data Pipeline Architecture
