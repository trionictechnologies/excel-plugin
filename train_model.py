#!/usr/bin/env python3
"""
Main Training Script
Train ledger classification models for levels 3 and 4
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config_loader import ConfigLoader
from data_preprocessor import DataPreprocessor
from trainer import ModelTrainer


def compute_class_weights(labels):
    """Compute class weights for imbalanced datasets"""
    unique_classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    return dict(zip(unique_classes, weights))


def train_classification_3(config, data_file):
    """Train classification level 3 model"""
    print("\n" + "="*80)
    print("TRAINING CLASSIFICATION LEVEL 3 MODEL")
    print("="*80 + "\n")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Load and prepare data
    print(f"Loading training data from: {data_file}")
    df = preprocessor.load_training_data(data_file, classification_level=3)
    print(f"Loaded {len(df)} records")
    
    # Prepare training data
    training_config = config.get_training_config()
    texts, labels, _ = preprocessor.prepare_training_data(
        df,
        classification_level=3,
        augment=training_config.get('augmentation', True)
    )
    
    print(f"Total training samples (after augmentation): {len(texts)}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Split data
    model_config = config.get_model_config()
    X_train, X_val, y_train, y_val, _, _ = preprocessor.split_data(
        texts,
        labels,
        test_size=model_config.get('validation_split', 0.2)
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Save encoders
    preprocessor.save_encoders()
    print("Label encoders saved")
    
    # Initialize trainer
    trainer = ModelTrainer(config, preprocessor, classification_level=3)
    
    # Create model
    num_classes = len(np.unique(labels))
    trainer.create_model(num_classes)
    
    # Prepare data for training
    X_train_prepared, y_train_prepared, X_val_prepared, y_val_prepared = trainer.prepare_data(
        X_train, y_train, X_val, y_val
    )
    
    # Compute class weights if enabled
    class_weights = None
    if training_config.get('use_class_weights', True):
        print("Computing class weights...")
        class_weights = compute_class_weights(y_train_prepared)
    
    # Train model
    history = trainer.train(
        X_train_prepared,
        y_train_prepared,
        X_val_prepared,
        y_val_prepared,
        class_weights=class_weights
    )
    
    # Evaluate
    class_names = preprocessor.get_class_names(3)
    results = trainer.evaluate(X_val_prepared, y_val_prepared, class_names)
    
    # Save model
    trainer.save_model()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Final Validation Accuracy: {results['test_accuracy']:.4f}")
    print(f"Final Validation Precision: {results['test_precision']:.4f}")
    print(f"Final Validation Recall: {results['test_recall']:.4f}")
    print("="*80 + "\n")
    
    return trainer, results


def train_classification_4(config, data_file):
    """Train classification level 4 model"""
    print("\n" + "="*80)
    print("TRAINING CLASSIFICATION LEVEL 4 MODEL")
    print("="*80 + "\n")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Load encoders from classification 3
    preprocessor.load_encoders()
    
    # Load and prepare data
    print(f"Loading training data from: {data_file}")
    df = preprocessor.load_training_data(data_file, classification_level=4)
    print(f"Loaded {len(df)} records")
    
    # Prepare training data
    training_config = config.get_training_config()
    texts, labels, context = preprocessor.prepare_training_data(
        df,
        classification_level=4,
        augment=training_config.get('augmentation', True)
    )
    
    print(f"Total training samples (after augmentation): {len(texts)}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Split data
    model_config = config.get_model_config()
    X_train, X_val, y_train, y_val, ctx_train, ctx_val = preprocessor.split_data(
        texts,
        labels,
        context,
        test_size=model_config.get('validation_split', 0.2)
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Save encoders
    preprocessor.save_encoders()
    print("Label encoders saved")
    
    # Initialize trainer
    trainer = ModelTrainer(config, preprocessor, classification_level=4)
    
    # Create model
    num_classes = len(np.unique(labels))
    num_context_classes = len(preprocessor.get_class_names(3))
    trainer.create_model(num_classes, num_context_classes)
    
    # Prepare data for training
    X_train_prepared, y_train_prepared, X_val_prepared, y_val_prepared = trainer.prepare_data(
        X_train, y_train, X_val, y_val, ctx_train, ctx_val
    )
    
    # Compute class weights if enabled
    class_weights = None
    if training_config.get('use_class_weights', True):
        print("Computing class weights...")
        class_weights = compute_class_weights(y_train_prepared)
    
    # Train model
    history = trainer.train(
        X_train_prepared,
        y_train_prepared,
        X_val_prepared,
        y_val_prepared,
        class_weights=class_weights
    )
    
    # Evaluate
    class_names = preprocessor.get_class_names(4)
    results = trainer.evaluate(X_val_prepared, y_val_prepared, class_names)
    
    # Save model
    trainer.save_model()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Final Validation Accuracy: {results['test_accuracy']:.4f}")
    print(f"Final Validation Precision: {results['test_precision']:.4f}")
    print(f"Final Validation Recall: {results['test_recall']:.4f}")
    print("="*80 + "\n")
    
    return trainer, results


def main():
    parser = argparse.ArgumentParser(description='Train Ledger Classification Model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data file (Excel/CSV)')
    parser.add_argument('--level', type=int, choices=[3, 4], required=True,
                       help='Classification level to train (3 or 4)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Validate data file
    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # Load configuration
    print("Loading configuration...")
    config = ConfigLoader(args.config)
    print("Configuration loaded successfully\n")
    
    # Train based on level
    try:
        if args.level == 3:
            trainer, results = train_classification_3(config, args.data)
        else:
            trainer, results = train_classification_4(config, args.data)
        
        print("\n✓ Training completed successfully!")
        print(f"Model saved to: {config.config['paths']['model_dir']}")
        print(f"Logs saved to: {config.config['paths']['logs_dir']}")
        
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
