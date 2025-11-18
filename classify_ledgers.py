#!/usr/bin/env python3
"""
Main Classification Script
Classify ledgers in Excel files using trained models
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config_loader import ConfigLoader
from classifier import LedgerClassifier
from excel_integration import ExcelHandler


def classify_file(input_file, output_file, classification_level, config_file='config.yaml'):
    """
    Classify ledgers in an Excel file
    
    Args:
        input_file: Input Excel file path
        output_file: Output Excel file path
        classification_level: Classification level (3 or 4)
        config_file: Configuration file path
    """
    print("\n" + "="*80)
    print(f"LEDGER CLASSIFICATION - LEVEL {classification_level}")
    print("="*80 + "\n")
    
    # Load configuration
    print("Loading configuration...")
    config = ConfigLoader(config_file)
    print("✓ Configuration loaded\n")
    
    # Initialize Excel handler
    print("Initializing Excel handler...")
    excel_handler = ExcelHandler(config)
    print("✓ Excel handler initialized\n")
    
    # Read input file
    print(f"Reading input file: {input_file}")
    df = excel_handler.read_excel(input_file)
    print(f"✓ Loaded {len(df)} records\n")
    
    # Prepare data for classification
    print("Preparing data for classification...")
    df, indices_to_classify = excel_handler.prepare_data_for_classification(
        df,
        classification_level
    )
    print(f"✓ Found {len(indices_to_classify)} ledgers to classify\n")
    
    if len(indices_to_classify) == 0:
        print("No ledgers to classify. All ledgers already have classifications.")
        return
    
    # Initialize classifier
    print(f"Loading classification model (Level {classification_level})...")
    classifier = LedgerClassifier(config, classification_level=classification_level)
    classifier.load_model()
    print("✓ Model loaded successfully\n")
    
    # Classify
    print("Classifying ledgers...")
    input_col = config.get_excel_config()['input_column']
    
    if classification_level == 3:
        results = classifier.classify_batch(
            df.loc[indices_to_classify, input_col].tolist()
        )
    else:
        # Level 4 requires classification 3
        results = classifier.classify_batch(
            df.loc[indices_to_classify, input_col].tolist(),
            df.loc[indices_to_classify, 'Classification 3'].tolist()
        )
    
    # Update DataFrame with results
    output_col = f'Classification {classification_level}'
    confidence_col = f'{output_col} Confidence'
    
    for idx, (pred_class, confidence) in zip(indices_to_classify, results):
        df.at[idx, output_col] = pred_class
        df.at[idx, confidence_col] = confidence
    
    print(f"✓ Classified {len(indices_to_classify)} ledgers\n")
    
    # Statistics
    confidence_threshold = config.get_excel_config().get('confidence_threshold', 0.7)
    high_conf = sum(1 for _, conf in results if conf >= confidence_threshold)
    low_conf = len(results) - high_conf
    
    print("Classification Statistics:")
    print(f"  - High confidence (≥{confidence_threshold:.0%}): {high_conf}")
    print(f"  - Low confidence (<{confidence_threshold:.0%}): {low_conf}")
    print(f"  - Average confidence: {sum(c for _, c in results) / len(results):.2%}\n")
    
    # Write output file
    print(f"Writing results to: {output_file}")
    excel_handler.write_classifications(
        df,
        output_file,
        classification_level,
        highlight_low_confidence=True
    )
    print("✓ Results saved\n")
    
    # Create report
    report_file = output_file.replace('.xlsx', '_report.xlsx')
    print(f"Creating classification report: {report_file}")
    excel_handler.create_classification_report(
        df,
        report_file,
        classification_level
    )
    print("✓ Report saved\n")
    
    print("="*80)
    print("CLASSIFICATION COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
    
    return df


def classify_text(ledger_name, classification_level, classification_3=None, 
                 top_k=3, config_file='config.yaml'):
    """
    Classify a single ledger name
    
    Args:
        ledger_name: Ledger name to classify
        classification_level: Classification level (3 or 4)
        classification_3: Classification 3 value (required for level 4)
        top_k: Number of top predictions to show
        config_file: Configuration file path
    """
    # Load configuration
    config = ConfigLoader(config_file)
    
    # Initialize classifier
    classifier = LedgerClassifier(config, classification_level=classification_level)
    classifier.load_model()
    
    # Get top predictions
    if classification_level == 4 and not classification_3:
        print("Error: classification_3 is required for level 4 classification")
        return
    
    top_predictions = classifier.get_top_predictions(
        ledger_name,
        top_k=top_k,
        classification_3=classification_3
    )
    
    print(f"\nLedger: {ledger_name}")
    if classification_3:
        print(f"Classification 3: {classification_3}")
    print(f"\nTop {top_k} Predictions (Level {classification_level}):")
    print("-" * 60)
    
    for i, (pred_class, confidence) in enumerate(top_predictions, 1):
        print(f"{i}. {pred_class:<40} {confidence:>6.2%}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Classify Ledgers using AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify an Excel file (Level 3)
  python classify_ledgers.py --input ledgers.xlsx --output classified.xlsx --level 3
  
  # Classify an Excel file (Level 4)
  python classify_ledgers.py --input ledgers.xlsx --output classified.xlsx --level 4
  
  # Classify a single ledger name
  python classify_ledgers.py --text "Purchase of Raw Material" --level 3
  
  # Get top 5 predictions for a ledger
  python classify_ledgers.py --text "Salaries and Wages" --level 3 --top-k 5
        """
    )
    
    parser.add_argument('--input', type=str,
                       help='Input Excel file path')
    parser.add_argument('--output', type=str,
                       help='Output Excel file path')
    parser.add_argument('--text', type=str,
                       help='Single ledger name to classify')
    parser.add_argument('--level', type=int, choices=[3, 4], required=True,
                       help='Classification level (3 or 4)')
    parser.add_argument('--class3', type=str,
                       help='Classification 3 value (required for level 4 when using --text)')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of top predictions to show (for --text mode)')
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
                args.top_k,
                args.config
            )
        
        elif args.input and args.output:
            # File classification
            if not Path(args.input).exists():
                print(f"Error: Input file not found: {args.input}")
                sys.exit(1)
            
            classify_file(
                args.input,
                args.output,
                args.level,
                args.config
            )
        
        else:
            parser.print_help()
            print("\nError: Either provide --input and --output, or provide --text")
            sys.exit(1)
        
        print("✓ Success!")
        
    except Exception as e:
        print(f"\n✗ Classification failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
