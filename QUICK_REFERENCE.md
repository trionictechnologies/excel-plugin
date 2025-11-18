# Quick Reference Card

## 🚀 Common Commands

### Training

```bash
# Train Level 3 classification
python train_model.py --data data/training_data.xlsx --level 3

# Train Level 4 classification  
python train_model.py --data data/training_data.xlsx --level 4
```

### Classification

```bash
# Classify Excel file (Level 3)
python classify_ledgers.py --input ledgers.xlsx --output result.xlsx --level 3

# Classify Excel file (Level 4)
python classify_ledgers.py --input ledgers.xlsx --output result.xlsx --level 4

# Test single ledger
python classify_ledgers.py --text "Purchase of Raw Material" --level 3

# Get top 5 predictions
python classify_ledgers.py --text "Employee Salaries" --level 3 --top-k 5
```

## 📁 Required File Format

### Training Data (Level 3)
| Ledger Name | Classification 3 |
|------------|-----------------|
| Purchase of Steel | Cost of Goods Sold |
| Employee Salaries | Employee Benefits Expenses |

### Training Data (Level 4)
| Ledger Name | Classification 3 | Classification 4 |
|------------|-----------------|------------------|
| Purchase of Steel | Cost of Goods Sold | Purchase of Raw Material |
| Office Salaries | Employee Benefits Expenses | Salaries and Wages |

### Input for Classification (Level 3)
| Ledger Name |
|------------|
| Purchase of Cotton |
| Employee Provident Fund |

### Input for Classification (Level 4)
| Ledger Name | Classification 3 |
|------------|-----------------|
| Purchase of Cotton | Cost of Goods Sold |
| Employee Provident Fund | Employee Benefits Expenses |

## ⚙️ Configuration Files

### config.yaml Location
```
/workspace/config.yaml
```

### Key Settings
```yaml
model:
  architecture: transformer    # or: lstm
  batch_size: 32              # reduce if out of memory
  epochs: 50                  # increase for better accuracy
  learning_rate: 0.001

excel:
  confidence_threshold: 0.7    # 70% confidence minimum
```

## 📊 Understanding Results

### Confidence Scores
- **90-100%**: Extremely confident ✓
- **70-89%**: Confident ✓
- **50-69%**: Review needed ⚠️
- **<50%**: Definitely review ❌

### Color Coding in Excel
- 🟢 **Green**: High confidence (≥70%)
- 🔴 **Red**: Low confidence (<70%)

## 🔧 Troubleshooting

### Model not found
```bash
# Solution: Train the model first
python train_model.py --data data/training.xlsx --level 3
```

### Out of memory
```yaml
# Edit config.yaml
model:
  batch_size: 16  # or 8
```

### Low accuracy
```
- Add more training data (aim for 100+ per category)
- Balance classes (equal examples per category)
- Increase epochs in config.yaml
```

### Column not found
```
- Check Excel column name: "Ledger Name" (exact spelling)
- Must be in first row
- Case sensitive
```

## 📂 File Locations

```
models/                    # Trained models saved here
logs/                      # Training logs and visualizations
data/                      # Put training data here
checkpoints/               # Model checkpoints during training
```

## 🎯 Workflow

1. **Prepare training data** (50-100 examples per category)
2. **Train model**: `python train_model.py --data data/train.xlsx --level 3`
3. **Check accuracy** in logs/ folder (aim for >90%)
4. **Classify**: `python classify_ledgers.py --input ledgers.xlsx --output result.xlsx --level 3`
5. **Review red highlighted** entries in result.xlsx
6. **Correct and retrain** with corrected data

## 💡 Best Practices

### Training Data
✅ 50-100 examples per category  
✅ Clear, consistent naming  
✅ No typos or special characters  
✅ Balanced across categories  
✅ Use actual ledger names from your books  

### Classification
✅ Review low confidence predictions  
✅ Keep threshold ≥ 70%  
✅ Retrain monthly with corrections  
✅ Check classification reports  

## 🔍 Python API

```python
from src.config_loader import ConfigLoader
from src.classifier import LedgerClassifier

# Load
config = ConfigLoader()
classifier = LedgerClassifier(config, classification_level=3)
classifier.load_model()

# Classify
prediction, confidence = classifier.classify_single("Purchase of Steel")
print(f"{prediction} ({confidence:.2%})")
```

## 📞 Getting Help

1. Check column names match exactly
2. Ensure training data has no blank rows
3. Verify model was trained successfully
4. Review logs/ folder for errors
5. Check USAGE_GUIDE.md for detailed help

## ⌨️ Keyboard Shortcuts (Jupyter)

- **Shift + Enter**: Run cell
- **Esc**: Command mode
- **A**: Insert cell above
- **B**: Insert cell below
- **DD**: Delete cell

## 📈 Monitoring Performance

### Check Training Success
```bash
# Look for these files after training:
models/ledger_classifier_level_3.h5
models/label_encoder_3.pkl
logs/run_YYYYMMDD_HHMMSS/confusion_matrix.png
logs/run_YYYYMMDD_HHMMSS/training_history.png
```

### Check Classification Quality
```bash
# Open after classification:
classified_ledgers.xlsx          # Results
classified_ledgers_report.xlsx   # Statistics
```

## 🎓 Classification Categories

### P&L Account - Level 3
- Cost of Goods Sold
- Employee Benefits Expenses
- Finance Costs
- Depreciation and Amortization Expense
- Other Expenses
- Trade Receivables
- Cash and Bank Balance
- Loans and Advances
- Other Assets
- Investments

### Cost of Goods Sold - Level 4
- Purchase of Raw Material
- Purchase of Packing Material
- Purchase of Other Material
- Purchase of Stock In Trade
- Opening Stock
- Direct Labour
- Factory Overheads

## 🔄 Update Cycle

### Monthly
- Review low-confidence predictions
- Collect corrections
- Add to training data
- Retrain model

### Quarterly
- Analyze classification patterns
- Update categories if needed
- Optimize configuration
- Performance review

## 💻 System Requirements

### Minimum
- Python 3.8+
- 4GB RAM
- 2GB disk space

### Recommended
- Python 3.10+
- 8GB RAM
- GPU (optional)
- 5GB disk space

## 🌐 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow; print('✓ Ready!')"
```

## 📝 Notes

- Training takes 5-30 minutes
- Classification is instant (<1 second per ledger)
- Models work offline (no internet needed)
- All data stays local (private and secure)

---

**Version**: 1.0.0  
**Print this card for quick reference!**

For detailed help, see:
- README.md - Complete documentation
- USAGE_GUIDE.md - Step-by-step instructions
- CONFIGURATION_GUIDE.md - All settings explained
