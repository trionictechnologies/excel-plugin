# System Architecture

This document describes the technical architecture of the Ledger Classification System.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│  (Excel Files, Command Line, Python API, Jupyter Notebook)  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                          │
│  ┌──────────────────┐         ┌─────────────────────┐       │
│  │  train_model.py  │         │ classify_ledgers.py │       │
│  │  (Training CLI)  │         │  (Inference CLI)    │       │
│  └──────────────────┘         └─────────────────────┘       │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      BUSINESS LOGIC LAYER                    │
│  ┌────────────┐  ┌──────────┐  ┌────────────┐              │
│  │  Trainer   │  │Classifier│  │Excel       │              │
│  │           │  │          │  │Integration │              │
│  └────────────┘  └──────────┘  └────────────┘              │
│  ┌────────────────────────────────────────────┐             │
│  │        Data Preprocessor                   │             │
│  │   (Cleaning, Augmentation, Encoding)       │             │
│  └────────────────────────────────────────────┘             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                       MODEL LAYER                            │
│  ┌──────────────┐  ┌──────────┐  ┌─────────────┐           │
│  │ Transformer  │  │   LSTM   │  │   Hybrid    │           │
│  │ Classifier   │  │Classifier│  │  Classifier │           │
│  └──────────────┘  └──────────┘  └─────────────┘           │
│                                                              │
│  ┌────────────────────────────────────────────┐             │
│  │    Sentence-BERT Embeddings (Pre-trained)  │             │
│  └────────────────────────────────────────────┘             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Training │  │  Models  │  │  Logs    │  │ Config   │   │
│  │   Data   │  │ & Weights│  │          │  │   YAML   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Description

### 1. User Interface Layer

**Purpose**: Provide multiple ways to interact with the system

**Components**:
- **Excel Files**: Primary input/output format for accountants
- **Command Line Interface**: For batch processing and automation
- **Python API**: For programmatic access and integration
- **Jupyter Notebook**: For interactive exploration and testing

### 2. Application Layer

**Purpose**: Coordinate high-level operations

#### train_model.py
- Orchestrates the training pipeline
- Handles command-line arguments
- Manages training workflow
- Reports progress and results

#### classify_ledgers.py
- Coordinates classification operations
- Supports single and batch classification
- Generates reports and statistics
- Handles Excel integration

### 3. Business Logic Layer

**Purpose**: Core business operations and data processing

#### Trainer (trainer.py)
- **Responsibilities**:
  - Model creation and compilation
  - Training loop management
  - Callbacks (early stopping, checkpointing)
  - Evaluation and metrics
  - Visualization generation
- **Key Methods**:
  - `create_model()`: Initialize model architecture
  - `train()`: Execute training
  - `evaluate()`: Generate metrics and reports
  - `save_model()`: Persist trained weights

#### Classifier (classifier.py)
- **Responsibilities**:
  - Model loading
  - Single and batch inference
  - Confidence scoring
  - Top-K predictions
- **Key Methods**:
  - `load_model()`: Load trained model
  - `classify_single()`: Predict for one ledger
  - `classify_batch()`: Predict for multiple ledgers
  - `get_top_predictions()`: Return top K classes

#### Excel Integration (excel_integration.py)
- **Responsibilities**:
  - Read/write Excel files
  - Data validation
  - Result formatting
  - Report generation
- **Key Methods**:
  - `read_excel()`: Load data
  - `write_classifications()`: Save results with formatting
  - `create_classification_report()`: Generate statistics
  - `export_training_data()`: Prepare data for retraining

#### Data Preprocessor (data_preprocessor.py)
- **Responsibilities**:
  - Text cleaning and normalization
  - Data augmentation
  - Label encoding
  - Train/validation splitting
- **Key Methods**:
  - `clean_ledger_name()`: Normalize text
  - `augment_text()`: Create variations
  - `prepare_training_data()`: Process dataset
  - `split_data()`: Create train/val sets

#### Config Loader (config_loader.py)
- **Responsibilities**:
  - Parse YAML configuration
  - Provide configuration access
  - Directory management
  - Hierarchy navigation
- **Key Methods**:
  - `get_classification_hierarchy()`: Access categories
  - `get_model_config()`: Get model settings
  - `get_classes_for_classification()`: Query hierarchy

### 4. Model Layer

**Purpose**: Deep learning architectures for classification

#### Transformer Classifier
- **Architecture**:
  - Pre-trained Sentence-BERT embeddings (384-dim)
  - Dense layers with batch normalization
  - Dropout for regularization
  - Softmax output
- **Advantages**:
  - Semantic understanding
  - Transfer learning benefits
  - Fast inference
  - Best accuracy

#### LSTM Classifier
- **Architecture**:
  - Trainable word embeddings
  - Bidirectional LSTM layers
  - Dense classification head
  - Softmax output
- **Advantages**:
  - Sequential pattern recognition
  - Customizable embeddings
  - Lower memory footprint

#### Hybrid Classifier
- **Architecture**:
  - Two-input model:
    1. Text embeddings (Sentence-BERT)
    2. Context embeddings (Classification 3)
  - Concatenation layer
  - Dense layers with regularization
  - Softmax output
- **Use Case**: Classification Level 4
- **Advantage**: Leverages hierarchical structure

### 5. Data Layer

**Purpose**: Persistent storage and configuration

#### File Structure
```
workspace/
├── data/              # Training data
│   ├── *.xlsx         # Excel training files
│   └── *.csv          # CSV training files
├── models/            # Trained models
│   ├── *.h5           # Model weights
│   └── *.pkl          # Encoders
├── logs/              # Training logs
│   └── run_*/         # Per-run logs
│       ├── *.png      # Visualizations
│       └── *.json     # Metrics
└── config.yaml        # Configuration
```

## Data Flow

### Training Pipeline

```
1. Load Config
   ↓
2. Load Training Data (Excel/CSV)
   ↓
3. Preprocess Text
   - Clean ledger names
   - Remove special characters
   - Normalize case
   ↓
4. Augment Data
   - Create variations
   - Expand dataset
   ↓
5. Encode Labels
   - Map classes to integers
   - Save encoders
   ↓
6. Split Data
   - Training set (80%)
   - Validation set (20%)
   ↓
7. Create Model
   - Choose architecture
   - Initialize weights
   ↓
8. Encode Text
   - Transformer: Sentence embeddings
   - LSTM: Tokenize and pad
   ↓
9. Train Model
   - Forward pass
   - Calculate loss
   - Backward pass
   - Update weights
   - Repeat for epochs
   ↓
10. Evaluate
    - Calculate metrics
    - Generate visualizations
    - Create confusion matrix
    ↓
11. Save Model
    - Save weights
    - Save encoders
    - Save config
```

### Classification Pipeline

```
1. Load Config
   ↓
2. Load Trained Model
   - Load weights
   - Load encoders
   ↓
3. Read Input Excel
   - Parse columns
   - Validate format
   ↓
4. Identify Ledgers to Classify
   - Non-empty Ledger Name
   - Empty Classification field
   ↓
5. Preprocess Text
   - Clean ledger names
   - Same as training
   ↓
6. Encode Text
   - Generate embeddings
   ↓
7. Model Inference
   - Forward pass
   - Get probabilities
   ↓
8. Decode Predictions
   - Convert indices to class names
   - Extract confidence scores
   ↓
9. Write Results
   - Add classification columns
   - Apply color coding
   - Save Excel
   ↓
10. Generate Report
    - Summary statistics
    - Class distribution
    - Low confidence list
```

## Key Design Patterns

### 1. Strategy Pattern
- Multiple model architectures (Transformer, LSTM, Hybrid)
- Selectable via configuration
- Consistent interface

### 2. Factory Pattern
- Model creation based on configuration
- Encoder creation (Label, Tokenizer)

### 3. Template Method Pattern
- Training pipeline with customizable steps
- Evaluation workflow

### 4. Dependency Injection
- ConfigLoader injected into all components
- Promotes testability and flexibility

## Technology Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow | ≥2.12.0 | Deep learning framework |
| Sentence-Transformers | ≥2.2.0 | Pre-trained embeddings |
| scikit-learn | ≥1.2.0 | ML utilities, metrics |
| pandas | ≥1.5.0 | Data manipulation |
| openpyxl | ≥3.1.0 | Excel file handling |
| numpy | ≥1.23.0 | Numerical operations |
| PyYAML | ≥6.0 | Configuration parsing |

### Model Architecture Details

#### Transformer Classifier (Default)

```python
Input: Text (string)
    ↓
Sentence-BERT Encoder (384-dim)
    ↓
Dense(256, relu) + Dropout(0.3) + BatchNorm
    ↓
Dense(128, relu) + Dropout(0.3) + BatchNorm
    ↓
Dense(64, relu) + Dropout(0.3)
    ↓
Dense(num_classes, softmax)
    ↓
Output: Class probabilities
```

**Parameters**: ~200K (excluding pre-trained encoder)

#### Hybrid Classifier (Level 4)

```python
Input 1: Text Embeddings (384-dim)
Input 2: Context Index (1-dim)
    ↓                        ↓
Dense(256, relu)      Embedding(64)
    ↓                        ↓
Dropout(0.3)          Dense(128, relu)
    ↓                        ↓
BatchNorm             Dropout(0.3)
    ↓                        ↓
    └────────┬───────────────┘
             ↓
        Concatenate (384-dim)
             ↓
    Dense(256, relu) + Dropout + BatchNorm
             ↓
    Dense(128, relu) + Dropout
             ↓
    Dense(num_classes, softmax)
             ↓
    Output: Class probabilities
```

**Parameters**: ~300K (excluding pre-trained encoder)

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Text Preprocessing | O(n) | n = text length |
| Sentence Embedding | O(n) | Constant for fixed-length |
| Model Inference | O(1) | Fixed architecture |
| Batch Classification | O(m) | m = batch size |

### Space Complexity

| Component | Size | Notes |
|-----------|------|-------|
| Sentence-BERT Model | ~80MB | Pre-trained weights |
| Classification Head | ~1-5MB | Depends on num_classes |
| Training Data | ~1KB/sample | Varies with text length |
| Model Checkpoints | ~5-10MB | Per checkpoint |

### Scalability

- **Training**: Scales linearly with dataset size
- **Inference**: Can process 1000+ ledgers/second on CPU
- **Memory**: Can train on 4GB RAM with batch_size=16
- **GPU**: Optional but speeds up training 5-10x

## Security Considerations

### Data Privacy
- Training data may contain sensitive financial information
- Models are stored locally (no cloud dependencies)
- No external API calls during classification

### Input Validation
- Excel file format validation
- Column name verification
- Data type checking
- SQL injection prevention (N/A - no database)

### Model Integrity
- Models saved with checksums
- Version tracking in config files
- Reproducible training with random seeds

## Extensibility

### Adding New Classification Levels

1. Update `config.yaml` with new hierarchy
2. Prepare training data with new columns
3. Train model with `--level N`
4. No code changes required

### Adding New Model Architectures

1. Create new model class in `models.py`
2. Implement required methods:
   - `__init__()`
   - `_build_model()`
   - `compile()`
3. Update `trainer.py` to handle new architecture
4. Add to config options

### Integration with Other Systems

The system can be integrated via:
- **Python API**: Import modules directly
- **Command Line**: Call scripts from other programs
- **REST API**: Wrap in Flask/FastAPI (not included)
- **Excel Add-in**: VBA wrapper (not included)

## Testing Strategy

### Unit Tests
- Test each module independently
- Mock dependencies
- Cover edge cases

### Integration Tests
- Test end-to-end workflows
- Verify data flow
- Check file I/O

### Validation Tests
- Test on held-out data
- Cross-validation
- Performance benchmarks

## Monitoring and Logging

### Training Logs
- TensorBoard integration
- Metrics per epoch
- Learning curves
- Confusion matrices

### Classification Logs
- Timestamp each operation
- Record confidence scores
- Track low-confidence predictions
- Audit trail for compliance

## Error Handling

### Graceful Degradation
- Handle missing files
- Validate data formats
- Catch and report errors
- Provide helpful messages

### Recovery Mechanisms
- Model checkpointing
- Resume training from checkpoint
- Fallback to default config
- Data validation before processing

---

**Version**: 1.0.0  
**Last Updated**: 2025-11-18  
**Architecture**: Modular, scalable, maintainable
