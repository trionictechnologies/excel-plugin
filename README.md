# AI-Powered Ledger Classification for Excel

An intelligent system that automatically classifies accounting ledgers into predefined categories using deep learning. The system thinks like a professional Chartered Accountant to accurately categorize ledgers based on their names.

## 🌟 Features

- **Multi-Level Classification**: Supports Classification Level 3 and Level 4
- **Deep Learning Powered**: Uses transformer-based models (Sentence-BERT) and LSTM architectures
- **Excel Integration**: Seamlessly works with Microsoft Excel files
- **Confidence Scoring**: Provides confidence scores for each prediction
- **Visual Feedback**: Color-coded results (green for high confidence, red for low confidence)
- **Flexible Configuration**: YAML-based configuration for easy customization
- **Hierarchical Classification**: Level 4 classification uses Level 3 context for better accuracy
- **Comprehensive Reports**: Generates detailed classification reports and statistics
- **Professional Accuracy**: Trained to think like a Chartered Accountant
- **Continuous Learning**: Easy to retrain with corrected classifications

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Classifying Ledgers](#classifying-ledgers)
- [Classification Hierarchy](#classification-hierarchy)
- [Model Architectures](#model-architectures)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (8GB recommended for training)
- GPU (optional, but recommended for faster training)

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import tensorflow; import transformers; print('Installation successful!')"
```

## ⚡ Quick Start

### 1. Prepare Your Training Data

Create an Excel file with the following columns:

**For Level 3:**
- `Ledger Name`: The name of the ledger account
- `Classification 3`: The category (e.g., "Cost of Goods Sold", "Employee Benefits Expenses")

**For Level 4:**
- `Ledger Name`: The name of the ledger account
- `Classification 3`: The Level 3 category
- `Classification 4`: The Level 4 sub-category (e.g., "Purchase of Raw Material", "Salaries and Wages")

Sample data files are provided in the `data/` directory.

### 2. Train the Models

**Train Level 3 Classification:**
```bash
python train_model.py --data data/sample_training_data_level3.xlsx --level 3
```

**Train Level 4 Classification:**
```bash
python train_model.py --data data/sample_training_data_level4.xlsx --level 4
```

Training will:
- Preprocess and augment your data
- Train the deep learning model
- Generate evaluation metrics and visualizations
- Save the trained model

### 3. Classify New Ledgers

**Classify an Excel File:**
```bash
python classify_ledgers.py --input your_ledgers.xlsx --output classified_ledgers.xlsx --level 3
```

**Classify a Single Ledger:**
```bash
python classify_ledgers.py --text "Purchase of Raw Material" --level 3
```

**Get Top 5 Predictions:**
```bash
python classify_ledgers.py --text "Employee Salaries" --level 3 --top-k 5
```

## 📁 Project Structure

```
ledger-classification/
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dependencies
├── train_model.py             # Training script
├── classify_ledgers.py        # Classification script
│
├── src/                       # Source code modules
│   ├── __init__.py
│   ├── config_loader.py       # Configuration management
│   ├── data_preprocessor.py   # Data preprocessing and cleaning
│   ├── models.py              # Deep learning model architectures
│   ├── trainer.py             # Training pipeline
│   ├── classifier.py          # Inference engine
│   └── excel_integration.py   # Excel file handling
│
├── data/                      # Training data
│   ├── sample_training_data_level3.xlsx
│   └── sample_training_data_level4.xlsx
│
├── models/                    # Saved models and encoders
│   ├── ledger_classifier_level_3.h5
│   ├── ledger_classifier_level_4.h5
│   ├── label_encoder_3.pkl
│   └── label_encoder_4.pkl
│
├── logs/                      # Training logs and visualizations
│   └── run_YYYYMMDD_HHMMSS/
│       ├── confusion_matrix.png
│       ├── training_history.png
│       └── evaluation_results.json
│
└── checkpoints/               # Model checkpoints during training
```

## ⚙️ Configuration

The `config.yaml` file controls all aspects of the system:

### Classification Hierarchy

Define your organization's classification structure:

```yaml
classification_hierarchy:
  P&L Account:
    classification_3:
      - Cost of Goods Sold
      - Employee Benefits Expenses
      - Finance Costs
      # ... more categories
    
    classification_4:
      Cost of Goods Sold:
        - Purchase of Raw Material
        - Purchase of Packing Material
        # ... more sub-categories
```

### Model Configuration

```yaml
model:
  architecture: transformer    # Options: transformer, lstm
  hidden_dim: 256
  dropout: 0.3
  learning_rate: 0.001
  batch_size: 32
  epochs: 50
```

### Transformer Settings

```yaml
transformer:
  pretrained_model: sentence-transformers/all-MiniLM-L6-v2
  max_length: 64
  fine_tune: true
```

### Excel Integration

```yaml
excel:
  input_column: Ledger Name
  confidence_threshold: 0.7
  review_low_confidence: true
```

## 📖 Usage

### Training Models

#### Basic Training

```bash
# Train Level 3
python train_model.py --data training_data.xlsx --level 3

# Train Level 4
python train_model.py --data training_data.xlsx --level 4
```

#### With Custom Configuration

```bash
python train_model.py --data training_data.xlsx --level 3 --config custom_config.yaml
```

#### What Happens During Training:

1. **Data Loading**: Reads your Excel/CSV file
2. **Preprocessing**: Cleans ledger names (lowercase, remove special characters)
3. **Augmentation**: Creates variations of each ledger name for better generalization
4. **Model Creation**: Builds the neural network architecture
5. **Training**: Trains the model with early stopping and learning rate scheduling
6. **Evaluation**: Generates metrics, confusion matrix, and training curves
7. **Saving**: Saves the trained model and encoders

### Classifying Ledgers

#### Classify Excel Files

```bash
python classify_ledgers.py \
  --input ledgers.xlsx \
  --output classified_ledgers.xlsx \
  --level 3
```

**Input Excel Format:**
- Must have a column named "Ledger Name"
- For Level 4, must also have "Classification 3" column

**Output:**
- Original data with added columns:
  - `Classification 3` (or 4)
  - `Classification 3 Confidence` (or 4)
- Color-coded results (green = high confidence, red = low confidence)
- Separate report file with statistics

#### Classify Single Ledger

```bash
# Level 3
python classify_ledgers.py --text "Purchase of Steel" --level 3

# Level 4 (requires Classification 3)
python classify_ledgers.py \
  --text "Purchase of Steel" \
  --level 4 \
  --class3 "Cost of Goods Sold"
```

#### Get Multiple Predictions

```bash
python classify_ledgers.py \
  --text "Employee Benefits" \
  --level 3 \
  --top-k 5
```

Output:
```
Ledger: Employee Benefits

Top 5 Predictions (Level 3):
------------------------------------------------------------
1. Employee Benefits Expenses                      95.23%
2. Other Expenses                                   3.45%
3. Cost of Goods Sold                               0.89%
4. Finance Costs                                    0.32%
5. Depreciation and Amortization Expense            0.11%
```

## 🏗️ Classification Hierarchy

### Classification Level 3 (Main Categories)

For **P&L Account**:
- Cost of Goods Sold
- Employee Benefits Expenses
- Finance Costs
- Depreciation and Amortization Expense
- Other Expenses
- Other Profit and Loss Items
- PPE & IA (Net)
- Investments
- Deferred Tax (Net)
- Loans and Advances
- Trade Receivables
- Cash and Bank Balance
- Other Assets

For **Balance Sheet**:
- Fixed Assets
- Current Assets
- Current Liabilities
- Non-Current Liabilities
- Equity and Reserves

### Classification Level 4 (Sub-Categories)

Each Level 3 category has multiple Level 4 sub-categories. For example:

**Cost of Goods Sold** → 
- Purchase of Raw Material
- Purchase of Packing Material
- Purchase of Other Material
- Opening Stock
- Direct Labour
- Factory Overheads

**Employee Benefits Expenses** →
- Salaries and Wages
- Staff Welfare
- Provident Fund Contribution
- Gratuity

## 🧠 Model Architectures

### Transformer-Based Classifier (Default)

- **Pre-trained Model**: Sentence-BERT (all-MiniLM-L6-v2)
- **Architecture**: Sentence embeddings → Dense layers → Classification
- **Advantages**:
  - Better semantic understanding
  - Pre-trained on large text corpus
  - Faster inference
  - Better with limited training data

### LSTM-Based Classifier

- **Architecture**: Embedding → Bidirectional LSTM → Dense → Classification
- **Advantages**:
  - Good for sequential patterns
  - Smaller model size
  - Customizable embeddings

### Hybrid Classifier (Level 4)

- **Architecture**: Text embeddings + Context embeddings → Concatenate → Classification
- **Special Feature**: Uses Classification 3 as context for better Level 4 predictions
- **Advantages**:
  - Hierarchical understanding
  - More accurate sub-category classification

## 🔧 Advanced Usage

### Programmatic Usage

```python
from src.config_loader import ConfigLoader
from src.classifier import LedgerClassifier

# Initialize
config = ConfigLoader('config.yaml')
classifier = LedgerClassifier(config, classification_level=3)
classifier.load_model()

# Classify single ledger
prediction, confidence = classifier.classify_single("Purchase of Raw Material")
print(f"Classification: {prediction} (Confidence: {confidence:.2%})")

# Classify batch
ledgers = ["Salaries", "Bank Interest", "Depreciation"]
results = classifier.classify_batch(ledgers)

for ledger, (pred, conf) in zip(ledgers, results):
    print(f"{ledger}: {pred} ({conf:.2%})")
```

### Custom Training Loop

```python
from src.config_loader import ConfigLoader
from src.data_preprocessor import DataPreprocessor
from src.trainer import ModelTrainer

# Setup
config = ConfigLoader()
preprocessor = DataPreprocessor(config)

# Load and prepare data
df = preprocessor.load_training_data('data.xlsx', classification_level=3)
texts, labels, _ = preprocessor.prepare_training_data(df, classification_level=3)
X_train, X_val, y_train, y_val, _, _ = preprocessor.split_data(texts, labels)

# Train
trainer = ModelTrainer(config, preprocessor, classification_level=3)
trainer.create_model(num_classes=len(set(labels)))
X_train_prep, y_train_prep, X_val_prep, y_val_prep = trainer.prepare_data(
    X_train, y_train, X_val, y_val
)
history = trainer.train(X_train_prep, y_train_prep, X_val_prep, y_val_prep)

# Evaluate and save
class_names = preprocessor.get_class_names(3)
results = trainer.evaluate(X_val_prep, y_val_prep, class_names)
trainer.save_model()
```

### Modifying Classification Categories

1. Edit `config.yaml` to add/remove categories
2. Prepare new training data with updated categories
3. Retrain the model:
```bash
python train_model.py --data updated_data.xlsx --level 3
```

### Adjusting Confidence Threshold

In `config.yaml`:
```yaml
excel:
  confidence_threshold: 0.8  # Higher = more conservative
```

Or programmatically:
```python
classifier.confidence_threshold = 0.8
```

## 🔍 Troubleshooting

### Common Issues

**1. Model not found error**
```
Solution: Train the model first using train_model.py
```

**2. Out of memory during training**
```
Solution: Reduce batch_size in config.yaml or use a smaller model
```

**3. Low accuracy**
```
Solutions:
- Collect more training data
- Ensure training data is balanced across classes
- Increase epochs or adjust learning rate
- Try different model architecture
```

**4. Excel file errors**
```
Solution: Ensure Excel file has correct column names:
- "Ledger Name" (exact spelling and capitalization)
- "Classification 3" or "Classification 4"
```

**5. GPU not being used**
```
Solution: Install tensorflow-gpu or check CUDA installation
```

### Getting Help

For issues or questions:
1. Check the configuration file syntax
2. Verify training data format
3. Review log files in `logs/` directory
4. Check model evaluation metrics

## 📊 Performance Metrics

The system provides comprehensive metrics:

- **Accuracy**: Overall correctness
- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of correctly identified positives
- **Confidence Score**: Model's certainty (0-1)
- **Confusion Matrix**: Visual representation of predictions
- **Class-wise Statistics**: Performance per category

## 🎯 Best Practices

1. **Training Data**:
   - Aim for at least 50-100 examples per category
   - Ensure balanced distribution across classes
   - Include variations in naming conventions
   - Review and clean data before training

2. **Model Selection**:
   - Use Transformer for better accuracy (default)
   - Use LSTM if you have limited resources
   - Always validate on held-out test data

3. **Classification**:
   - Review low-confidence predictions manually
   - Periodically retrain with corrected predictions
   - Keep confidence threshold ≥ 0.7 for quality

4. **Production Usage**:
   - Backup your trained models
   - Version your configuration files
   - Log all classifications for audit trail
   - Monitor classification distribution

## 🔄 Continuous Improvement

To improve model performance over time:

1. **Collect Feedback**:
```python
# Export corrected classifications as new training data
excel_handler.export_training_data(df, 'new_training_data.xlsx', level=3)
```

2. **Retrain Periodically**:
```bash
# Combine old and new data
python train_model.py --data combined_data.xlsx --level 3
```

3. **Monitor Performance**:
- Track confidence scores over time
- Identify frequently misclassified ledgers
- Update configuration for edge cases

## 📝 License

This project is provided as-is for accounting and ledger classification purposes.

## 🙏 Acknowledgments

- Built with TensorFlow and Sentence-Transformers
- Uses state-of-the-art NLP techniques
- Designed for Chartered Accountants and finance professionals

---

**Version**: 1.0.0  
**Last Updated**: 2025-11-18

For technical support or feature requests, please refer to the project documentation.
