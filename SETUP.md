# 🚀 Quick Setup Guide

Get started in **5 minutes**!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `openai` - OpenAI API client
- `pandas` - Data handling
- `openpyxl` - Excel support
- `pyyaml` - Configuration
- `python-dotenv` - Environment variables
- `tqdm` - Progress bars

## Step 2: Get OpenAI API Key

1. Go to **[platform.openai.com](https://platform.openai.com)**
2. Sign up or login
3. Navigate to **API Keys** section
4. Click **"Create new secret key"**
5. Copy your key (starts with `sk-...`)

**Cost**: ~$0.15 per 1000 ledgers with gpt-4o-mini

## Step 3: Configure API Key

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your key
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4o-mini
```

Or set as environment variable:
```bash
export OPENAI_API_KEY=sk-your-key-here
```

## Step 4: Test the Setup

```bash
python classify_ledgers.py --text "Purchase of Raw Material" --level 3
```

Expected output:
```
🤖 AI Ledger Classifier (gpt-4o-mini)
══════════════════════════════════════════════════════════════════════

📝 Ledger: Purchase of Raw Material

🎯 Classification (Level 3):
──────────────────────────────────────────────────────────────────────
Category: Cost of Goods Sold
Confidence: 95.0%
Reasoning: Raw materials are directly used in manufacturing...
```

✅ **Success!** Your setup is complete.

## Step 5: Try with Sample Data

Classify the sample file:

```bash
python classify_ledgers.py \
  --input data/sample_ledgers.csv \
  --output data/classified_sample.xlsx \
  --level 3
```

Check the output:
- Open `data/classified_sample.xlsx`
- See classifications with confidence scores
- Green = high confidence, Red = review needed

## Step 6: Customize for Your Organization

Edit `config.yaml` to match your categories:

```yaml
classification_hierarchy:
  P&L Account:
    classification_3:
      - Cost of Goods Sold
      - Employee Benefits Expenses
      - Your Custom Category    # Add here
```

## 🎯 You're Ready!

Now you can:

1. **Classify your ledgers**:
   ```bash
   python classify_ledgers.py \
     --input your_ledgers.xlsx \
     --output results.xlsx \
     --level 3
   ```

2. **Test individual ledgers**:
   ```bash
   python classify_ledgers.py \
     --text "Your Ledger Name" \
     --level 3
   ```

3. **Get multiple suggestions**:
   ```bash
   python classify_ledgers.py \
     --text "Your Ledger Name" \
     --level 3 \
     --alternatives
   ```

## 🔧 Troubleshooting

### "OpenAI API key not found"
- Create `.env` file in project root
- Add: `OPENAI_API_KEY=your-key`

### "Module not found" error
- Run: `pip install -r requirements.txt`

### "Rate limit exceeded"
- Wait 60 seconds
- Or upgrade your OpenAI plan

### Excel column error
- Ensure column is named exactly: `Ledger Name`

## 💰 Costs

| Model | Cost per 1000 | Accuracy | Speed |
|-------|--------------|----------|--------|
| gpt-4o-mini | $0.15 | 90-95% | Fast |
| gpt-4o | $2.50 | 95-98% | Medium |

**Recommendation**: Start with `gpt-4o-mini`

## 📚 Next Steps

- Read **README.md** for complete documentation
- Check **USAGE_GUIDE.md** for detailed examples
- Customize **config.yaml** for your organization

## ✅ Checklist

- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Got OpenAI API key
- [ ] Created `.env` file with API key
- [ ] Tested with sample command
- [ ] Classified sample data file
- [ ] Reviewed `config.yaml`
- [ ] Ready to classify your ledgers!

---

**Estimated setup time**: 5 minutes  
**Total cost to test**: < $0.01  
**Time saved**: Hours of manual classification ⏰

**Questions?** Check README.md or USAGE_GUIDE.md

**Let's automate your ledger classification! 🚀**
