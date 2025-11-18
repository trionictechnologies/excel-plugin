# Project Index - Ledger Classification System

Complete index of all files and their purposes.

## 📚 Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| **README.md** | Main documentation with quick start | Everyone |
| **USAGE_GUIDE.md** | Step-by-step guide for accountants | End Users |
| **CONFIGURATION_GUIDE.md** | Detailed configuration reference | Power Users |
| **ARCHITECTURE.md** | Technical architecture details | Developers |
| **PROJECT_SUMMARY.md** | Complete project overview | Managers |
| **QUICK_REFERENCE.md** | One-page command reference | Everyone |
| **INDEX.md** | This file - project index | Everyone |

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| **config.yaml** | Main configuration (categories, model settings) |
| **requirements.txt** | Python dependencies |
| **.gitignore** | Git ignore patterns |

## 🚀 Executable Scripts

| File | Purpose | Usage |
|------|---------|-------|
| **train_model.py** | Train classification models | `python train_model.py --data data.xlsx --level 3` |
| **classify_ledgers.py** | Classify ledgers | `python classify_ledgers.py --input in.xlsx --output out.xlsx --level 3` |

## 📦 Source Code Modules

| File | Purpose | Key Classes/Functions |
|------|---------|---------------------|
| **src/__init__.py** | Package initialization | - |
| **src/config_loader.py** | Configuration management | `ConfigLoader` |
| **src/data_preprocessor.py** | Data preprocessing | `DataPreprocessor` |
| **src/models.py** | Deep learning models | `TransformerClassifier`, `LSTMClassifier`, `HybridClassifier` |
| **src/trainer.py** | Training pipeline | `ModelTrainer` |
| **src/classifier.py** | Inference engine | `LedgerClassifier` |
| **src/excel_integration.py** | Excel file handling | `ExcelHandler` |

## 📊 Data Files

### Sample Data
| File | Purpose | Format |
|------|---------|--------|
| **data/sample_training_data_level3.csv** | Example Level 3 training data | CSV, 2 columns |
| **data/sample_training_data_level4.csv** | Example Level 4 training data | CSV, 3 columns |
| **data/README.md** | Data directory documentation | Markdown |

### User Data (You Create)
- `data/your_training_data.xlsx` - Your training examples
- `data/your_ledgers.xlsx` - Ledgers to classify

## 🧠 Model Files (Generated After Training)

| File | Created By | Contains |
|------|-----------|----------|
| **models/ledger_classifier_level_3.h5** | Training script | Level 3 model weights |
| **models/ledger_classifier_level_4.h5** | Training script | Level 4 model weights |
| **models/label_encoder_3.pkl** | Training script | Level 3 label encoder |
| **models/label_encoder_4.pkl** | Training script | Level 4 label encoder |
| **models/architecture_level_3.json** | Training script | Model architecture |
| **models/training_config_level_3.json** | Training script | Training configuration |

## 📈 Output Files (Generated)

### Training Outputs
Located in `logs/run_YYYYMMDD_HHMMSS/`:
- **confusion_matrix.png** - Visual confusion matrix
- **training_history.png** - Training curves (loss, accuracy, etc.)
- **evaluation_results.json** - Metrics in JSON format

### Classification Outputs
- **classified_ledgers.xlsx** - Input file with classifications added
- **classified_ledgers_report.xlsx** - Summary statistics

### Checkpoints
Located in `checkpoints/`:
- **class_3_best.h5** - Best model during Level 3 training
- **class_4_best.h5** - Best model during Level 4 training

## 📓 Interactive Notebooks

| File | Purpose |
|------|---------|
| **notebooks/QuickStart.ipynb** | Interactive tutorial and examples |

## 📂 Directory Structure

```
workspace/
├── 📄 Documentation (7 files)
│   ├── README.md
│   ├── USAGE_GUIDE.md
│   ├── CONFIGURATION_GUIDE.md
│   ├── ARCHITECTURE.md
│   ├── PROJECT_SUMMARY.md
│   ├── QUICK_REFERENCE.md
│   └── INDEX.md
│
├── ⚙️  Configuration (3 files)
│   ├── config.yaml
│   ├── requirements.txt
│   └── .gitignore
│
├── 🚀 Scripts (2 files)
│   ├── train_model.py
│   └── classify_ledgers.py
│
├── 📦 src/ - Source Code (7 files)
│   ├── __init__.py
│   ├── config_loader.py
│   ├── data_preprocessor.py
│   ├── models.py
│   ├── trainer.py
│   ├── classifier.py
│   └── excel_integration.py
│
├── 📊 data/ - Training Data
│   ├── sample_training_data_level3.csv
│   ├── sample_training_data_level4.csv
│   ├── README.md
│   └── (your data files)
│
├── 🧠 models/ - Trained Models (created after training)
│   ├── *.h5 files (model weights)
│   ├── *.pkl files (encoders)
│   └── *.json files (configs)
│
├── 📈 logs/ - Training Logs (created during training)
│   └── run_*/
│       ├── *.png (visualizations)
│       └── *.json (metrics)
│
├── 💾 checkpoints/ - Model Checkpoints (created during training)
│   └── *.h5 files
│
└── 📓 notebooks/ - Jupyter Notebooks
    └── QuickStart.ipynb
```

## 🎯 File Size Reference

| Category | Approximate Size |
|----------|-----------------|
| Documentation | ~500 KB |
| Source Code | ~150 KB |
| Sample Data | ~50 KB |
| Pre-trained Embeddings | ~80 MB (downloaded on first use) |
| Trained Models | ~5-10 MB per model |
| Training Logs | ~5-10 MB per run |

## 📖 Reading Order for New Users

### For Accountants (Non-Technical)
1. **README.md** - Overview and quick start
2. **USAGE_GUIDE.md** - Step-by-step instructions
3. **QUICK_REFERENCE.md** - Command cheat sheet
4. **data/README.md** - How to prepare training data

### For Power Users
1. **README.md** - Overview
2. **CONFIGURATION_GUIDE.md** - Customize settings
3. **QUICK_REFERENCE.md** - Quick commands
4. **notebooks/QuickStart.ipynb** - Interactive examples

### For Developers
1. **README.md** - Overview
2. **ARCHITECTURE.md** - System design
3. **PROJECT_SUMMARY.md** - Complete feature list
4. **Source code** in `src/` - Implementation details

### For Managers/Decision Makers
1. **PROJECT_SUMMARY.md** - What it does and why
2. **README.md** - Features and capabilities
3. **ARCHITECTURE.md** - Technical details (optional)

## 🔍 Finding Information

### How do I...

**...get started quickly?**
→ README.md → Quick Start section

**...train a model?**
→ USAGE_GUIDE.md → Training the System section
→ QUICK_REFERENCE.md → Training commands

**...classify ledgers?**
→ USAGE_GUIDE.md → Classifying Your Ledgers section
→ QUICK_REFERENCE.md → Classification commands

**...change configuration?**
→ CONFIGURATION_GUIDE.md → Complete reference
→ config.yaml → Edit directly

**...understand the architecture?**
→ ARCHITECTURE.md → Complete technical details
→ PROJECT_SUMMARY.md → High-level overview

**...prepare training data?**
→ data/README.md → Data preparation guide
→ USAGE_GUIDE.md → Preparing Your Data section

**...troubleshoot issues?**
→ USAGE_GUIDE.md → Troubleshooting section
→ README.md → Troubleshooting section
→ QUICK_REFERENCE.md → Common issues

**...customize categories?**
→ CONFIGURATION_GUIDE.md → Classification Hierarchy
→ config.yaml → Edit classification_hierarchy

**...integrate with my system?**
→ ARCHITECTURE.md → Integration section
→ README.md → Advanced Usage section

**...understand the code?**
→ ARCHITECTURE.md → Component descriptions
→ Source code (well-commented)

## 📞 Support Resources

### Documentation
- All `.md` files in root directory
- `data/README.md` for data preparation
- Comments in source code

### Examples
- Sample data in `data/` directory
- Jupyter notebook in `notebooks/`
- Example commands in QUICK_REFERENCE.md

### Configuration
- `config.yaml` with inline comments
- CONFIGURATION_GUIDE.md for detailed explanations
- Example configurations for different scenarios

## 🔄 Version Control

### Track These Files
✅ All source code (`.py` files)
✅ Documentation (`.md` files)
✅ Configuration (`config.yaml`)
✅ Requirements (`requirements.txt`)
✅ Sample data

### Don't Track (in .gitignore)
❌ Trained models (`models/*.h5`)
❌ Training logs (`logs/`)
❌ Checkpoints (`checkpoints/`)
❌ Your actual training data (optional)
❌ Python cache (`__pycache__/`)

## 🎓 Learning Path

### Beginner
1. Read README.md overview
2. Follow USAGE_GUIDE.md step-by-step
3. Try with sample data
4. Use QUICK_REFERENCE.md as cheat sheet

### Intermediate
1. Understand CONFIGURATION_GUIDE.md
2. Experiment with different settings
3. Explore notebooks/QuickStart.ipynb
4. Try Python API

### Advanced
1. Study ARCHITECTURE.md
2. Read source code
3. Customize models
4. Integrate with other systems

## 📊 File Dependencies

```
train_model.py
    ↓ imports
    ├── src/config_loader.py
    ├── src/data_preprocessor.py
    ├── src/trainer.py
    └── src/models.py
        ↓ uses
        ├── config.yaml
        └── data/*.csv or *.xlsx

classify_ledgers.py
    ↓ imports
    ├── src/config_loader.py
    ├── src/classifier.py
    ├── src/excel_integration.py
    └── src/models.py
        ↓ uses
        ├── config.yaml
        ├── models/*.h5
        └── models/*.pkl
```

## ✅ Verification Checklist

After downloading/cloning, verify you have:

- [ ] All 7 documentation files
- [ ] 2 executable scripts (train_model.py, classify_ledgers.py)
- [ ] 7 source code files in src/
- [ ] config.yaml and requirements.txt
- [ ] Sample data in data/ directory
- [ ] Jupyter notebook in notebooks/
- [ ] Empty directories (models/, logs/, checkpoints/)

## 🎯 Quick Start Paths

```
Installation → USAGE_GUIDE.md → Step 1
Training    → USAGE_GUIDE.md → Step 3
Classification → USAGE_GUIDE.md → Step 4
Troubleshooting → QUICK_REFERENCE.md → Troubleshooting
API Usage   → README.md → Advanced Usage
```

---

**This index is your map to the project. Bookmark it!**

**Last Updated**: 2025-11-18  
**Version**: 1.0.0  
**Total Files**: 30+ (excluding generated files)
