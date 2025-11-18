# Ledger Classification System - Usage Guide

## Step-by-Step Guide for Accountants

This guide will help you use the AI-powered ledger classification system without any programming knowledge.

## Table of Contents
1. [Installation](#installation)
2. [Preparing Your Data](#preparing-your-data)
3. [Training the System](#training-the-system)
4. [Classifying Your Ledgers](#classifying-your-ledgers)
5. [Understanding Results](#understanding-results)
6. [Tips for Best Results](#tips-for-best-results)

---

## 1. Installation

### Step 1: Install Python
1. Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Restart your computer

### Step 2: Install Required Libraries
Open Command Prompt (Windows) or Terminal (Mac/Linux) and run:
```bash
pip install -r requirements.txt
```

This will install all necessary components. It may take 10-15 minutes.

---

## 2. Preparing Your Data

### For Training (Classification Level 3)

Create an Excel file with these columns:

| Ledger Name | Classification 3 |
|------------|-----------------|
| Purchase of Raw Materials | Cost of Goods Sold |
| Salaries and Wages | Employee Benefits Expenses |
| Bank Interest | Finance Costs |
| Rent Expenses | Other Expenses |

**Requirements:**
- At least 50-100 examples per category
- Clear, consistent naming
- No duplicate entries
- No blank rows

### For Training (Classification Level 4)

Create an Excel file with these columns:

| Ledger Name | Classification 3 | Classification 4 |
|------------|-----------------|------------------|
| Purchase of Steel | Cost of Goods Sold | Purchase of Raw Material |
| Purchase of Boxes | Cost of Goods Sold | Purchase of Packing Material |
| Factory Wages | Cost of Goods Sold | Direct Labour |
| Office Staff Salary | Employee Benefits Expenses | Salaries and Wages |

### For Classification (Your Ledgers)

Your Excel file should have at least:

| Ledger Name |
|------------|
| Purchase of Cotton |
| Employee Provident Fund |
| Depreciation - Machinery |
| ... |

For Level 4 classification, also include:

| Ledger Name | Classification 3 |
|------------|-----------------|
| Purchase of Cotton | Cost of Goods Sold |
| Employee Provident Fund | Employee Benefits Expenses |
| ... | ... |

---

## 3. Training the System

### Training Level 3 Classification

1. **Open Command Prompt/Terminal**
2. **Navigate to the project folder:**
   ```bash
   cd path/to/ledger-classification
   ```

3. **Run the training command:**
   ```bash
   python train_model.py --data data/my_training_data.xlsx --level 3
   ```

4. **Wait for training to complete** (10-30 minutes depending on data size)

5. **Check the results:**
   - Look for "Training completed successfully"
   - Note the accuracy percentage
   - Check the `logs/` folder for visualizations

### Training Level 4 Classification

**Important:** Train Level 3 first, then Level 4!

```bash
python train_model.py --data data/my_training_data_level4.xlsx --level 4
```

### What to Expect During Training

You'll see output like:
```
Loading configuration...
Loading training data from: data/my_training_data.xlsx
Loaded 500 records
Total training samples: 750
Training set: 600 samples
Validation set: 150 samples

Epoch 1/50
━━━━━━━━━━━━━━━━━━━━ 19s 32ms/step - accuracy: 0.6234
...
Epoch 25/50
━━━━━━━━━━━━━━━━━━━━ 18s 30ms/step - accuracy: 0.9456

Final Validation Accuracy: 0.9456
```

---

## 4. Classifying Your Ledgers

### Classify an Entire Excel File

1. **Prepare your Excel file** with ledger names

2. **Run the classification command:**

   **For Level 3:**
   ```bash
   python classify_ledgers.py --input my_ledgers.xlsx --output classified_ledgers.xlsx --level 3
   ```

   **For Level 4:**
   ```bash
   python classify_ledgers.py --input my_ledgers.xlsx --output classified_ledgers.xlsx --level 4
   ```

3. **Find your results:**
   - `classified_ledgers.xlsx`: Your data with classifications
   - `classified_ledgers_report.xlsx`: Detailed statistics

### Classify a Single Ledger (Quick Test)

```bash
python classify_ledgers.py --text "Purchase of Raw Material" --level 3
```

Output:
```
Ledger: Purchase of Raw Material

Top 3 Predictions (Level 3):
------------------------------------------------------------
1. Cost of Goods Sold                              97.34%
2. Other Expenses                                   1.89%
3. PPE & IA (Net)                                   0.45%
```

### Get Top 5 Suggestions

```bash
python classify_ledgers.py --text "Employee Insurance" --level 3 --top-k 5
```

---

## 5. Understanding Results

### Output Excel File

Your classified Excel will have these new columns:

| Ledger Name | Classification 3 | Classification 3 Confidence | Classification 3 Review |
|------------|-----------------|----------------------------|----------------------|
| Purchase of Steel | Cost of Goods Sold | 0.95 | FALSE |
| Some Unclear Entry | Other Expenses | 0.45 | TRUE |

**Color Coding:**
- 🟢 **Green**: High confidence (≥70%) - Safe to accept
- 🔴 **Red**: Low confidence (<70%) - Review manually

### Report File

The report includes:
1. **Summary**: Total ledgers, classified count, average confidence
2. **Class Distribution**: How many ledgers in each category
3. **Confidence by Class**: Average confidence for each category
4. **Low Confidence**: List of ledgers needing review

### Confidence Scores Explained

- **90-100%**: Extremely confident - Almost certainly correct
- **70-89%**: Confident - Likely correct, quick review recommended
- **50-69%**: Uncertain - Manual review required
- **<50%**: Very uncertain - Definitely needs manual review

---

## 6. Tips for Best Results

### Training Data Quality

✅ **DO:**
- Include variations: "Purchase of Raw Material", "Raw Material Purchase", "Raw Materials Bought"
- Cover all categories equally (50-100 examples each)
- Use actual ledger names from your books
- Keep consistent terminology
- Clean up typos and special characters

❌ **DON'T:**
- Use too short names: "Purchase", "Expense" (too generic)
- Mix languages: Keep all English or all in one language
- Include account codes: "1001 - Purchase" (remove codes)
- Have spelling errors or inconsistencies

### Improving Accuracy

**If accuracy is low (<85%):**

1. **Add more training data** - Aim for 100+ examples per category
2. **Balance your classes** - Equal representation of all categories
3. **Review misclassifications** - Add these to training data with correct labels
4. **Check for ambiguous names** - Make ledger names more descriptive
5. **Increase training epochs** - Edit `config.yaml` and change `epochs: 50` to `epochs: 100`

**If some categories are always wrong:**
- Those categories might need more distinct examples
- Consider combining similar categories
- Add more varied examples for those categories

### Classification Best Practices

1. **Start with Level 3** - Get it working well before attempting Level 4
2. **Review low confidence predictions** - Always check red-highlighted entries
3. **Retrain periodically** - As you correct predictions, add them to training data
4. **Keep a backup** - Save your training data and trained models
5. **Test with known data** - Classify ledgers you already categorized to verify accuracy

### Common Issues and Solutions

**Issue: "Model not found"**
- **Solution**: Train the model first using `train_model.py`

**Issue: "Column 'Ledger Name' not found"**
- **Solution**: Ensure your Excel has exactly "Ledger Name" as column header (case-sensitive)

**Issue: Low accuracy on your data**
- **Solution**: Training data should match your actual ledger naming patterns

**Issue: All predictions have low confidence**
- **Solution**: Model needs more diverse training data

**Issue: Out of memory error**
- **Solution**: Reduce batch size in `config.yaml` from 32 to 16 or 8

---

## Quick Reference Card

### Train Model
```bash
python train_model.py --data training_data.xlsx --level 3
```

### Classify File
```bash
python classify_ledgers.py --input ledgers.xlsx --output result.xlsx --level 3
```

### Test Single Ledger
```bash
python classify_ledgers.py --text "Your Ledger Name" --level 3
```

### Get Help
```bash
python classify_ledgers.py --help
```

---

## Example Workflow

**Day 1: Setup and Training**
1. Prepare training data (200 ledgers with classifications)
2. Train Level 3: `python train_model.py --data training.xlsx --level 3`
3. Check accuracy in logs (aim for >90%)
4. If accuracy low, add more training data and retrain

**Day 2: Initial Classification**
1. Export all your ledgers to Excel
2. Classify: `python classify_ledgers.py --input all_ledgers.xlsx --output classified.xlsx --level 3`
3. Open `classified.xlsx` and review red-highlighted entries
4. Correct any mistakes

**Week 2: Level 4 Training**
1. Prepare Level 4 training data
2. Train Level 4: `python train_model.py --data training_l4.xlsx --level 4`
3. Classify: `python classify_ledgers.py --input all_ledgers.xlsx --output classified_l4.xlsx --level 4`
4. Review and correct

**Monthly: Continuous Improvement**
1. Collect all corrected classifications
2. Add to training data
3. Retrain models
4. Classification becomes more accurate over time

---

## Support and Troubleshooting

### Before Asking for Help:
1. ✓ Check column names match exactly
2. ✓ Ensure training data has no blank rows
3. ✓ Verify model was trained successfully
4. ✓ Check Excel file is not corrupted

### Log Files to Check:
- `logs/` folder: Contains training metrics and visualizations
- Check `training_history.png` to see if model learned properly
- Check `confusion_matrix.png` to see which categories are confused

---

**Remember**: The AI learns from your data. Better training data = Better classifications!

Good luck with your ledger classification! 🎯
