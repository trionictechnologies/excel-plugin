# Training Data Directory

This directory contains sample training data and is where you should place your own training data files.

## Sample Files

### sample_training_data_level3.csv
Sample training data for Classification Level 3 with ~80 examples across multiple categories.

**Format:**
- Column 1: `Ledger Name` - The ledger account name
- Column 2: `Classification 3` - The target classification category

**Categories included:**
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

### sample_training_data_level4.csv
Sample training data for Classification Level 4 with examples of hierarchical classification.

**Format:**
- Column 1: `Ledger Name` - The ledger account name
- Column 2: `Classification 3` - The Level 3 classification
- Column 3: `Classification 4` - The Level 4 sub-category

## Preparing Your Own Training Data

### Requirements

1. **File Format**: Excel (.xlsx) or CSV (.csv)
2. **Column Names**: Must match exactly (case-sensitive):
   - `Ledger Name`
   - `Classification 3` (for Level 3)
   - `Classification 4` (for Level 4, also needs Classification 3)

3. **Data Quality**:
   - No blank rows
   - No duplicate entries
   - Consistent terminology
   - At least 50-100 examples per category

### Example Excel Structure

**Level 3:**
```
Ledger Name              | Classification 3
-------------------------|---------------------------
Purchase of Steel        | Cost of Goods Sold
Employee Salaries        | Employee Benefits Expenses
Bank Interest           | Finance Costs
```

**Level 4:**
```
Ledger Name              | Classification 3              | Classification 4
-------------------------|-------------------------------|---------------------------
Purchase of Steel        | Cost of Goods Sold           | Purchase of Raw Material
Opening Stock            | Cost of Goods Sold           | Opening Stock
Factory Wages           | Cost of Goods Sold           | Direct Labour
Office Salaries         | Employee Benefits Expenses    | Salaries and Wages
```

## Data Collection Tips

### Getting Training Data

1. **Export from your accounting software**
   - Export ledger master with account names
   - Manually classify 50-100 examples per category
   - Save as Excel or CSV

2. **Historical data**
   - Use previously classified ledgers
   - Review and correct any misclassifications
   - Ensure consistency

3. **Team collaboration**
   - Have multiple accountants classify
   - Review disagreements
   - Maintain classification guidelines

### Balancing Your Dataset

Aim for roughly equal representation across categories:

**Good:**
```
Cost of Goods Sold:              100 examples
Employee Benefits Expenses:       95 examples
Finance Costs:                    90 examples
Other Expenses:                  105 examples
```

**Avoid:**
```
Cost of Goods Sold:              250 examples
Employee Benefits Expenses:       10 examples  ❌ Too few
Finance Costs:                     5 examples  ❌ Too few
Other Expenses:                  150 examples
```

### Quality Checks

Before training:

1. ✓ Check for typos in ledger names
2. ✓ Ensure classification categories match config.yaml
3. ✓ Remove any test/dummy entries
4. ✓ Verify no blank classifications
5. ✓ Check for consistent naming conventions

## Training Commands

### Train Level 3
```bash
python train_model.py --data data/your_data.xlsx --level 3
```

### Train Level 4
```bash
python train_model.py --data data/your_data_level4.xlsx --level 4
```

## Data Privacy

**Important**: Your training data may contain sensitive information.

- Keep training data secure
- Don't share with unauthorized persons
- Consider anonymizing ledger names if sharing
- Follow your organization's data policies

## Adding More Training Data

To improve model accuracy over time:

1. **Collect misclassifications**
   - Review low-confidence predictions
   - Correct any mistakes
   
2. **Export and combine**
   - Export corrected classifications
   - Combine with existing training data
   
3. **Retrain**
   - Run training again with expanded dataset
   - Model will improve with more data

Example workflow:
```bash
# Initial training
python train_model.py --data data/initial_data.xlsx --level 3

# Classify ledgers
python classify_ledgers.py --input all_ledgers.xlsx --output classified.xlsx --level 3

# Review and correct in Excel
# Save corrections as new_corrections.xlsx

# Combine data (in Excel or Python)
# combined_data.xlsx = initial_data.xlsx + new_corrections.xlsx

# Retrain with more data
python train_model.py --data data/combined_data.xlsx --level 3
```

## File Organization

Suggested organization:
```
data/
├── sample_training_data_level3.csv
├── sample_training_data_level4.csv
├── my_training_data_v1.xlsx
├── my_training_data_v2.xlsx
├── corrections_2024_01.xlsx
└── combined_training_data.xlsx
```

Keep versions for audit trail and reproducibility.

---

**Need Help?** See USAGE_GUIDE.md and README.md for detailed instructions.
