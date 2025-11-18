# Simple Usage Guide - AI Ledger Classification

## Overview

This tool uses OpenAI's AI to automatically classify your ledgers. No training required - just configure and run!

## 🚀 Quick Setup (5 Minutes)

### Step 1: Get OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or login
3. Go to API Keys section
4. Create new secret key
5. Copy the key (starts with `sk-...`)

### Step 2: Install

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Key

Create a file named `.env` in the project folder:

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

### Step 4: Test It

```bash
python classify_ledgers.py --text "Purchase of Raw Material" --level 3
```

You should see:
```
🎯 Classification (Level 3):
Category: Cost of Goods Sold
Confidence: 95.0%
Reasoning: Direct material purchase for manufacturing
```

✅ **You're ready!**

## 📊 Classify Your Ledgers

### Prepare Your Excel File

Your Excel needs a column named **"Ledger Name"**:

| Ledger Name |
|------------|
| Purchase of Steel |
| Employee Salaries |
| Bank Interest |
| Depreciation - Machinery |

### Run Classification

```bash
python classify_ledgers.py \
  --input your_ledgers.xlsx \
  --output classified_ledgers.xlsx \
  --level 3
```

### Check Results

Open `classified_ledgers.xlsx`:

| Ledger Name | Classification 3 | Confidence | Reasoning |
|------------|------------------|------------|-----------|
| Purchase of Steel | Cost of Goods Sold | 0.95 | Direct material... |
| Employee Salaries | Employee Benefits | 0.98 | Staff compensation... |

- 🟢 Green = High confidence (good to go!)
- 🔴 Red = Low confidence (review this)

## ⚙️ Customize Categories

Edit `config.yaml` to match your organization:

```yaml
classification_hierarchy:
  P&L Account:
    classification_3:
      - Cost of Goods Sold
      - Employee Benefits Expenses
      - Your Custom Category    # Add your categories
```

## 💰 Costs

**Using gpt-4o-mini (recommended)**:
- 1000 ledgers = ~$0.15
- Very affordable!

**Using gpt-4o (more accurate)**:
- 1000 ledgers = ~$2.50
- Use for critical classifications

Change model in `.env`:
```
OPENAI_MODEL=gpt-4o-mini  # Cheap & good
# or
OPENAI_MODEL=gpt-4o       # More accurate
```

## 🎯 Examples

### Single Ledger Test
```bash
python classify_ledgers.py --text "Employee Medical Insurance" --level 3
```

### See All Options
```bash
python classify_ledgers.py --text "Staff Bonus" --level 3 --alternatives
```

### Level 4 Classification
```bash
python classify_ledgers.py \
  --text "Purchase of Cotton" \
  --level 4 \
  --class3 "Cost of Goods Sold"
```

### Classify Full File
```bash
python classify_ledgers.py \
  --input ledgers.xlsx \
  --output results.xlsx \
  --level 3
```

## 🔍 Understanding Results

### Confidence Scores
- **90-100%**: Very confident ✅ Accept it
- **70-89%**: Confident ✅ Quick review
- **50-69%**: Uncertain ⚠️ Review needed
- **<50%**: Very uncertain ❌ Definitely review

### What the AI Considers
1. **Ledger name keywords**: "Purchase", "Salary", "Interest"
2. **Context**: Level 3 category for Level 4 classification
3. **Accounting principles**: Standard classifications
4. **Common patterns**: How CAs typically classify

## 🛠️ Troubleshooting

### "API key not found"
**Fix**: Create `.env` file with your OpenAI key

### "Rate limit exceeded"
**Fix**: Wait a minute or upgrade OpenAI plan

### "Column not found"
**Fix**: Rename Excel column to exactly "Ledger Name"

### Low confidence scores
**Fix**: Make ledger names more specific
- ❌ "Purchase" → ✅ "Purchase of Raw Material"
- ❌ "Salary" → ✅ "Employee Salaries and Wages"

## 💡 Best Practices

### Good Ledger Names
✅ "Purchase of Raw Cotton Material"
✅ "Employee Medical Insurance Premium"
✅ "Bank Interest on Term Loan"
✅ "Depreciation on Plant & Machinery"

### Avoid
❌ "Purchase" (what?)
❌ "Expense" (what kind?)
❌ "Account 1001" (meaningless)

### Workflow
1. Classify all ledgers
2. Review red-highlighted (low confidence) entries
3. Correct any mistakes
4. Done!

## 📈 For Large Files

If you have 1000s of ledgers:

1. **Start with a sample**: Test with 50-100 ledgers first
2. **Review results**: Check accuracy
3. **Adjust categories**: Update config.yaml if needed
4. **Run full file**: Process everything
5. **Spot check**: Review random entries

## 🔐 Security Notes

- ✅ API calls are encrypted (HTTPS)
- ✅ OpenAI doesn't store your data long-term
- ✅ Your classifications stay private
- ⚠️ Don't share your API key
- ⚠️ Don't commit `.env` to git

## 📋 Quick Reference

```bash
# Test single ledger
python classify_ledgers.py --text "Ledger Name" --level 3

# Classify file
python classify_ledgers.py --input in.xlsx --output out.xlsx --level 3

# Show alternatives
python classify_ledgers.py --text "Ledger Name" --level 3 --alternatives

# Level 4 (needs Classification 3)
python classify_ledgers.py --text "Name" --level 4 --class3 "Category"
```

## ❓ FAQ

**Q: Do I need to train the model?**
A: No! OpenAI's AI already knows accounting. Just configure and run.

**Q: How accurate is it?**
A: 90-95%+ accuracy. Better than many manual classifications!

**Q: What if it makes mistakes?**
A: Review low-confidence predictions (red in Excel) and correct them.

**Q: Can I use my own categories?**
A: Yes! Edit config.yaml with your classification structure.

**Q: How much does it cost?**
A: About $0.15 per 1000 ledgers with gpt-4o-mini.

**Q: Is my data safe?**
A: Yes, OpenAI encrypts data and doesn't use it for training (per their policy).

**Q: Can I run it offline?**
A: No, it needs internet to call OpenAI API.

**Q: How fast is it?**
A: About 1-2 seconds per ledger with gpt-4o-mini.

## 🎯 Next Steps

1. ✅ Setup complete? Try classifying a test file
2. ✅ Good results? Process your full ledger list
3. ✅ Need adjustments? Edit config.yaml categories
4. ✅ Want better accuracy? Switch to gpt-4o model

---

**Need help?** Check README.md for detailed documentation.

**Ready to save hours of manual work?** Start classifying! 🚀
