#!/usr/bin/env python3
"""
AI Ledger Classification Tool
Uses OpenAI API to classify ledgers like a Chartered Accountant
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config_loader import ConfigLoader
from ledger_agent import LedgerClassificationAgent
from excel_integration import ExcelHandler


def classify_file(input_file: str, 
                 output_file: str, 
                 classification_level: int,
                 config_file: str = 'config.yaml'):
    """
    Classify ledgers in an Excel file using AI agent
    
    Args:
        input_file: Input Excel file path
        output_file: Output Excel file path
        classification_level: Classification level (3 or 4)
        config_file: Configuration file path
    """
    print("\n" + "="*80)
    print(f"🤖 AI LEDGER CLASSIFICATION - LEVEL {classification_level}")
    print("="*80 + "\n")
    
    # Load configuration
    print("📋 Loading configuration...")
    config = ConfigLoader(config_file)
    print("✓ Configuration loaded\n")
    
    # Initialize agent
    print("🤖 Initializing AI Agent...")
    agent = LedgerClassificationAgent(config)
    print(f"✓ Using OpenAI model: {agent.model}\n")
    
    # Initialize Excel handler
    excel_handler = ExcelHandler(config)
    
    # Read input file
    print(f"📂 Reading input file: {input_file}")
    df = excel_handler.read_excel(input_file)
    print(f"✓ Loaded {len(df)} records\n")
    
    # Prepare data for classification
    print("🔍 Identifying ledgers to classify...")
    df, indices_to_classify = excel_handler.prepare_data_for_classification(
        df, classification_level
    )
    print(f"✓ Found {len(indices_to_classify)} ledgers to classify\n")
    
    if len(indices_to_classify) == 0:
        print("ℹ️  No ledgers to classify. All already have classifications.")
        return
    
    # Get ledgers to classify
    input_col = config.get_excel_config()['input_column']
    ledgers_to_classify = df.loc[indices_to_classify, input_col].tolist()
    
    # For level 4, also get classification 3
    classification_3_list = None
    if classification_level == 4:
        classification_3_list = df.loc[indices_to_classify, 'Classification 3'].tolist()
    
    # Classify using AI agent
    print("🧠 Classifying ledgers with AI...\n")
    
    output_col = f'Classification {classification_level}'
    confidence_col = f'{output_col} Confidence'
    reasoning_col = f'{output_col} Reasoning'
    
    # Add reasoning column if not exists
    if reasoning_col not in df.columns:
        df[reasoning_col] = ''
    
    # Process each ledger with progress bar
    with tqdm(total=len(indices_to_classify), desc="Classifying") as pbar:
        for i, idx in enumerate(indices_to_classify):
            ledger_name = ledgers_to_classify[i]
            classification_3 = classification_3_list[i] if classification_3_list else None
            
            try:
                classification, confidence, reasoning = agent.classify_single(
                    ledger_name,
                    classification_level=classification_level,
                    classification_3=classification_3
                )
                
                df.at[idx, output_col] = classification
                df.at[idx, confidence_col] = confidence
                df.at[idx, reasoning_col] = reasoning
                
            except Exception as e:
                print(f"\n⚠️  Error classifying '{ledger_name}': {e}")
                df.at[idx, output_col] = "Error"
                df.at[idx, confidence_col] = 0.0
                df.at[idx, reasoning_col] = str(e)
            
            pbar.update(1)
    
    print(f"\n✓ Classified {len(indices_to_classify)} ledgers\n")
    
    # Calculate statistics
    confidences = df.loc[indices_to_classify, confidence_col].tolist()
    confidence_threshold = config.get_excel_config().get('confidence_threshold', 0.7)
    high_conf = sum(1 for c in confidences if c >= confidence_threshold)
    low_conf = len(confidences) - high_conf
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    print("📊 Classification Statistics:")
    print(f"  ✓ High confidence (≥{confidence_threshold:.0%}): {high_conf}")
    print(f"  ⚠️  Low confidence (<{confidence_threshold:.0%}): {low_conf}")
    print(f"  📈 Average confidence: {avg_conf:.2%}\n")
    
    # Write output file
    print(f"💾 Writing results to: {output_file}")
    excel_handler.write_classifications(
        df, output_file, classification_level, highlight_low_confidence=True
    )
    print("✓ Results saved\n")
    
    # Create report
    report_file = output_file.replace('.xlsx', '_report.xlsx')
    print(f"📊 Creating classification report: {report_file}")
    excel_handler.create_classification_report(df, report_file, classification_level)
    print("✓ Report saved\n")
    
    print("="*80)
    print("✅ CLASSIFICATION COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")


def classify_text(ledger_name: str,
                 classification_level: int,
                 classification_3: str = None,
                 show_alternatives: bool = False,
                 config_file: str = 'config.yaml'):
    """
    Classify a single ledger name
    
    Args:
        ledger_name: Ledger name to classify
        classification_level: Classification level (3 or 4)
        classification_3: Classification 3 value (required for level 4)
        show_alternatives: Show alternative classifications
        config_file: Configuration file path
    """
    # Load configuration
    config = ConfigLoader(config_file)
    
    # Initialize agent
    agent = LedgerClassificationAgent(config)
    
    print(f"\n🤖 AI Ledger Classifier ({agent.model})")
    print("="*70)
    
    if show_alternatives:
        # Get top alternatives
        results = agent.classify_with_alternatives(
            ledger_name,
            classification_level=classification_level,
            classification_3=classification_3,
            top_k=5
        )
        
        print(f"\n📝 Ledger: {ledger_name}")
        if classification_3:
            print(f"📁 Classification 3: {classification_3}")
        print(f"\n🎯 Top Predictions (Level {classification_level}):")
        print("-" * 70)
        
        for i, result in enumerate(results, 1):
            conf = result.get('confidence', 0)
            bar = '█' * int(conf * 40)
            print(f"\n{i}. {result['classification']}")
            print(f"   Confidence: {conf:.1%} {bar}")
            print(f"   Reasoning: {result['reasoning']}")
    
    else:
        # Single classification
        classification, confidence, reasoning = agent.classify_single(
            ledger_name,
            classification_level=classification_level,
            classification_3=classification_3
        )
        
        print(f"\n📝 Ledger: {ledger_name}")
        if classification_3:
            print(f"📁 Classification 3: {classification_3}")
        print(f"\n🎯 Classification (Level {classification_level}):")
        print("-" * 70)
        print(f"Category: {classification}")
        print(f"Confidence: {confidence:.1%}")
        print(f"Reasoning: {reasoning}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='AI-Powered Ledger Classification using OpenAI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify an Excel file (Level 3)
  python classify_ledgers.py --input ledgers.xlsx --output classified.xlsx --level 3
  
  # Classify an Excel file (Level 4)
  python classify_ledgers.py --input ledgers.xlsx --output classified.xlsx --level 4
  
  # Classify a single ledger
  python classify_ledgers.py --text "Purchase of Raw Material" --level 3
  
  # Get alternative classifications
  python classify_ledgers.py --text "Employee Salaries" --level 3 --alternatives
  
  # Classify with context (Level 4)
  python classify_ledgers.py --text "Opening Stock" --level 4 --class3 "Cost of Goods Sold"

Note: Set OPENAI_API_KEY in .env file before running
        """
    )
    
    parser.add_argument('--input', type=str, help='Input Excel file path')
    parser.add_argument('--output', type=str, help='Output Excel file path')
    parser.add_argument('--text', type=str, help='Single ledger name to classify')
    parser.add_argument('--level', type=int, choices=[3, 4], required=True,
                       help='Classification level (3 or 4)')
    parser.add_argument('--class3', type=str,
                       help='Classification 3 value (required for level 4 when using --text)')
    parser.add_argument('--alternatives', action='store_true',
                       help='Show alternative classifications (for --text mode)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        if args.text:
            # Single text classification
            classify_text(
                args.text,
                args.level,
                args.class3,
                args.alternatives,
                args.config
            )
        
        elif args.input and args.output:
            # File classification
            if not Path(args.input).exists():
                print(f"❌ Error: Input file not found: {args.input}")
                sys.exit(1)
            
            classify_file(args.input, args.output, args.level, args.config)
        
        else:
            parser.print_help()
            print("\n❌ Error: Either provide --input and --output, or provide --text")
            sys.exit(1)
        
        print("✅ Success!")
        
    except Exception as e:
        print(f"\n❌ Classification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
