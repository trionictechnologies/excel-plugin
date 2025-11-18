# Configuration Guide

This guide explains all configuration options in `config.yaml` and how to customize the system for your organization.

## Table of Contents
1. [Classification Hierarchy](#classification-hierarchy)
2. [Model Configuration](#model-configuration)
3. [Training Configuration](#training-configuration)
4. [Excel Integration](#excel-integration)
5. [Advanced Settings](#advanced-settings)

---

## Classification Hierarchy

Define your organization's classification structure. The system supports multiple Level 1 categories (e.g., P&L Account, Balance Sheet), each with their own Level 3 and Level 4 classifications.

### Structure

```yaml
classification_hierarchy:
  P&L Account:                    # Level 1 category
    classification_3:             # Level 3 categories
      - Cost of Goods Sold
      - Employee Benefits Expenses
      - Finance Costs
      # ... more categories
    
    classification_4:             # Level 4 sub-categories
      Cost of Goods Sold:         # Level 3 category name
        - Purchase of Raw Material
        - Purchase of Packing Material
        - Opening Stock
        # ... more sub-categories
      
      Employee Benefits Expenses:
        - Salaries and Wages
        - Staff Welfare
        # ... more sub-categories
```

### How to Customize

**Adding a new Level 3 category:**

```yaml
classification_hierarchy:
  P&L Account:
    classification_3:
      - Cost of Goods Sold
      - Your New Category Name    # Add here
```

**Adding Level 4 sub-categories:**

```yaml
classification_4:
  Your New Category Name:
    - Sub-category 1
    - Sub-category 2
    - Sub-category 3
```

**Adding a new Level 1 category:**

```yaml
classification_hierarchy:
  P&L Account:
    # ... existing structure
  
  Balance Sheet:               # New Level 1 category
    classification_3:
      - Fixed Assets
      - Current Assets
      # ... more
```

### Best Practices

- ✅ Use clear, descriptive names
- ✅ Keep consistent terminology
- ✅ Follow accounting standards (IndAS, IFRS, GAAP)
- ❌ Don't use abbreviations unless industry-standard
- ❌ Avoid overlapping categories

---

## Model Configuration

Controls the deep learning model architecture and training parameters.

### Available Settings

```yaml
model:
  name: ledger_classifier          # Model identifier
  architecture: transformer        # Model type: transformer, lstm
  embedding_dim: 128              # Embedding dimension (LSTM only)
  hidden_dim: 256                 # Hidden layer size
  num_layers: 3                   # Number of layers (LSTM only)
  dropout: 0.3                    # Dropout rate (0.0 - 0.5)
  learning_rate: 0.001            # Initial learning rate
  batch_size: 32                  # Training batch size
  epochs: 50                      # Maximum training epochs
  validation_split: 0.2           # Validation data proportion
  early_stopping_patience: 10     # Epochs to wait before stopping
```

### Detailed Explanations

#### Architecture
- **`transformer`**: Uses pre-trained Sentence-BERT embeddings
  - Better accuracy
  - Faster training
  - Requires more memory
  - **Recommended for most cases**

- **`lstm`**: Uses custom word embeddings with LSTM
  - Lower memory usage
  - Customizable embeddings
  - Good for sequential patterns

#### Hidden Dimension
- Controls model capacity
- **Larger values** (512, 768):
  - More complex patterns
  - Requires more data
  - Slower training
- **Smaller values** (128, 256):
  - Faster training
  - Less overfitting
  - Good for smaller datasets

#### Dropout
- Prevents overfitting
- **0.0**: No dropout (may overfit)
- **0.3**: Moderate regularization (recommended)
- **0.5**: Strong regularization (for very small datasets)

#### Learning Rate
- Controls training speed
- **0.001**: Standard (recommended)
- **0.0001**: Slower, more stable
- **0.01**: Faster, but may be unstable

#### Batch Size
- Number of samples per training step
- **8-16**: Small batches (less memory, slower)
- **32**: Standard (recommended)
- **64-128**: Large batches (more memory, faster)
- Reduce if you get out-of-memory errors

#### Epochs
- Maximum training iterations
- **30-50**: Standard
- **100+**: For large datasets
- Early stopping will stop before this if model stops improving

### Performance Tuning

**For better accuracy:**
```yaml
hidden_dim: 512
dropout: 0.2
epochs: 100
```

**For faster training:**
```yaml
architecture: transformer
batch_size: 64
epochs: 30
```

**For limited memory:**
```yaml
architecture: lstm
batch_size: 16
hidden_dim: 128
```

**For small datasets (<500 samples):**
```yaml
dropout: 0.4
hidden_dim: 128
epochs: 100
early_stopping_patience: 15
```

---

## Training Configuration

Controls the training process and optimization.

### Available Settings

```yaml
training:
  optimizer: adam                 # Optimizer: adam, sgd, rmsprop
  loss_function: sparse_categorical_crossentropy
  metrics:
    - accuracy
    - precision
    - recall
  use_class_weights: true        # Balance imbalanced classes
  augmentation: true             # Data augmentation
```

### Detailed Explanations

#### Optimizer
- **`adam`**: Adaptive learning rate (recommended)
- **`sgd`**: Stochastic Gradient Descent (simpler, may need tuning)
- **`rmsprop`**: Good for recurrent networks

#### Class Weights
- **`true`**: Automatically balance classes
  - Use when categories have unequal samples
  - Prevents bias toward common categories
- **`false`**: No balancing
  - Use when classes are already balanced

#### Data Augmentation
- **`true`**: Creates variations of training data
  - Improves generalization
  - Recommended for small datasets
- **`false`**: Uses data as-is
  - Faster training
  - Use for very large datasets (>5000 samples)

---

## Transformer Configuration

Settings specific to transformer-based models.

```yaml
transformer:
  pretrained_model: sentence-transformers/all-MiniLM-L6-v2
  max_length: 64
  fine_tune: true
  freeze_base: false
```

### Available Pre-trained Models

#### Recommended Models

1. **`sentence-transformers/all-MiniLM-L6-v2`** (Default)
   - Size: 80MB
   - Speed: Fast
   - Accuracy: Good
   - **Best for: General use**

2. **`sentence-transformers/all-mpnet-base-v2`**
   - Size: 420MB
   - Speed: Medium
   - Accuracy: Excellent
   - **Best for: Maximum accuracy**

3. **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**
   - Size: 420MB
   - Speed: Medium
   - Multilingual: Yes
   - **Best for: Non-English ledgers**

4. **`sentence-transformers/all-MiniLM-L12-v2`**
   - Size: 120MB
   - Speed: Fast
   - Accuracy: Very Good
   - **Best for: Balance of speed and accuracy**

### Parameters

#### max_length
- Maximum text length in tokens
- **32**: Short texts only
- **64**: Standard (recommended)
- **128**: Long descriptions
- Longer = more memory

#### fine_tune
- **`true`**: Update embeddings during training
  - Better accuracy
  - Longer training
- **`false`**: Freeze embeddings
  - Faster training
  - Use for small datasets

---

## Excel Integration

Controls Excel file reading and writing.

```yaml
excel:
  input_column: Ledger Name           # Column with ledger names
  output_columns:                     # Output column names
    - Classification 3
    - Classification 4
  confidence_threshold: 0.7           # Confidence threshold (0.0 - 1.0)
  review_low_confidence: true         # Highlight low confidence
```

### Customization

#### Change Input Column Name

If your Excel uses different column names:

```yaml
excel:
  input_column: Account Name    # Instead of "Ledger Name"
```

#### Adjust Confidence Threshold

Controls what's considered "low confidence":

```yaml
confidence_threshold: 0.8    # Stricter (90%+ = high confidence)
confidence_threshold: 0.6    # Lenient (60%+ = high confidence)
```

**Recommendations:**
- **0.7**: Standard (70% confidence)
- **0.8**: Conservative (review more predictions)
- **0.6**: Aggressive (accept more predictions)

---

## File Paths

Configure where files are stored.

```yaml
paths:
  model_dir: models              # Trained models
  data_dir: data                 # Training data
  logs_dir: logs                 # Training logs
  checkpoints_dir: checkpoints   # Model checkpoints
```

### Absolute Paths

You can use absolute paths:

```yaml
paths:
  model_dir: /home/user/accounting/models
  data_dir: D:/Accounting/TrainingData
```

---

## Complete Configuration Templates

### For Small Organization (<1000 ledgers)

```yaml
model:
  architecture: transformer
  hidden_dim: 128
  dropout: 0.3
  learning_rate: 0.001
  batch_size: 16
  epochs: 50
  early_stopping_patience: 10

transformer:
  pretrained_model: sentence-transformers/all-MiniLM-L6-v2
  max_length: 64

training:
  use_class_weights: true
  augmentation: true

excel:
  confidence_threshold: 0.7
```

### For Large Organization (>5000 ledgers)

```yaml
model:
  architecture: transformer
  hidden_dim: 512
  dropout: 0.2
  learning_rate: 0.001
  batch_size: 64
  epochs: 100
  early_stopping_patience: 15

transformer:
  pretrained_model: sentence-transformers/all-mpnet-base-v2
  max_length: 128

training:
  use_class_weights: true
  augmentation: false    # Enough data without augmentation

excel:
  confidence_threshold: 0.8
```

### For Limited Hardware (Low RAM/No GPU)

```yaml
model:
  architecture: transformer
  hidden_dim: 128
  dropout: 0.3
  learning_rate: 0.001
  batch_size: 8
  epochs: 50

transformer:
  pretrained_model: sentence-transformers/all-MiniLM-L6-v2
  max_length: 32
  freeze_base: true     # Don't fine-tune embeddings

training:
  augmentation: false   # Reduce memory usage
```

### For Maximum Accuracy (Research/Audit)

```yaml
model:
  architecture: transformer
  hidden_dim: 768
  dropout: 0.2
  learning_rate: 0.0005
  batch_size: 32
  epochs: 200
  early_stopping_patience: 25

transformer:
  pretrained_model: sentence-transformers/all-mpnet-base-v2
  max_length: 128
  fine_tune: true

training:
  use_class_weights: true
  augmentation: true

excel:
  confidence_threshold: 0.9    # Only accept very high confidence
```

---

## Troubleshooting Configuration Issues

### Issue: Out of Memory

**Solution:**
```yaml
model:
  batch_size: 8          # Reduce from 32
  hidden_dim: 128        # Reduce from 256

transformer:
  max_length: 32         # Reduce from 64
```

### Issue: Training Too Slow

**Solution:**
```yaml
model:
  batch_size: 64         # Increase from 32
  epochs: 30             # Reduce from 50

transformer:
  freeze_base: true      # Don't fine-tune
```

### Issue: Low Accuracy

**Solution:**
```yaml
model:
  hidden_dim: 512        # Increase from 256
  epochs: 100            # Increase from 50
  dropout: 0.2           # Reduce from 0.3

training:
  augmentation: true     # Enable augmentation
```

### Issue: Overfitting (High training accuracy, low validation accuracy)

**Solution:**
```yaml
model:
  dropout: 0.5           # Increase from 0.3
  hidden_dim: 128        # Reduce from 256

training:
  augmentation: true     # Add more variation
```

---

## Validation

After modifying `config.yaml`, validate it:

```bash
python -c "from src.config_loader import ConfigLoader; c = ConfigLoader(); print('✓ Configuration valid!')"
```

---

## Examples of Configuration Changes

### Example 1: Adding Industry-Specific Categories

**Manufacturing Company:**
```yaml
classification_hierarchy:
  P&L Account:
    classification_3:
      - Direct Material Cost
      - Direct Labour Cost
      - Manufacturing Overhead
      - Administrative Expenses
      - Selling and Distribution
```

**Service Company:**
```yaml
classification_hierarchy:
  P&L Account:
    classification_3:
      - Professional Fees Income
      - Consultant Charges
      - Employee Costs
      - Office and Admin
      - Marketing and Sales
```

### Example 2: Multilingual Support

For Hindi/English mixed ledgers:

```yaml
transformer:
  pretrained_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  max_length: 128    # Longer for mixed scripts
```

### Example 3: Quick Testing Configuration

For rapid iteration:

```yaml
model:
  batch_size: 64
  epochs: 10         # Quick training
  early_stopping_patience: 3

transformer:
  freeze_base: true  # Don't fine-tune
```

---

**Remember**: After changing configuration, you need to retrain your models for changes to take effect!

```bash
python train_model.py --data your_data.xlsx --level 3
```
