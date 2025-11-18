"""
Training Pipeline Module
Handles model training, evaluation, and checkpointing
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from tqdm import tqdm

from config_loader import ConfigLoader
from data_preprocessor import DataPreprocessor
from models import TransformerClassifier, LSTMClassifier, HybridClassifier


class ModelTrainer:
    """Handles training and evaluation of classification models"""
    
    def __init__(self, 
                 config_loader: ConfigLoader,
                 preprocessor: DataPreprocessor,
                 classification_level: int = 3):
        """
        Initialize model trainer
        
        Args:
            config_loader: ConfigLoader instance
            preprocessor: DataPreprocessor instance
            classification_level: Target classification level (3 or 4)
        """
        self.config = config_loader
        self.preprocessor = preprocessor
        self.classification_level = classification_level
        self.model = None
        self.history = None
        
        # Create directories
        self.model_dir = Path(self.config.config['paths']['model_dir'])
        self.logs_dir = Path(self.config.config['paths']['logs_dir'])
        self.checkpoints_dir = Path(self.config.config['paths']['checkpoints_dir'])
        
        for dir_path in [self.model_dir, self.logs_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup timestamp for this training run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.logs_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
    
    def create_model(self, 
                    num_classes: int,
                    num_context_classes: Optional[int] = None) -> None:
        """
        Create model based on configuration
        
        Args:
            num_classes: Number of output classes
            num_context_classes: Number of context classes (for hybrid model)
        """
        model_config = self.config.get_model_config()
        transformer_config = self.config.get_transformer_config()
        
        architecture = model_config.get('architecture', 'transformer')
        
        if architecture == 'transformer':
            if self.classification_level == 4 and num_context_classes:
                print("Creating Hybrid Transformer model for Classification 4...")
                self.model = HybridClassifier(
                    num_classes=num_classes,
                    num_context_classes=num_context_classes,
                    pretrained_model=transformer_config['pretrained_model'],
                    hidden_dim=model_config['hidden_dim'],
                    dropout=model_config['dropout']
                )
            else:
                print("Creating Transformer model...")
                self.model = TransformerClassifier(
                    num_classes=num_classes,
                    pretrained_model=transformer_config['pretrained_model'],
                    hidden_dim=model_config['hidden_dim'],
                    dropout=model_config['dropout'],
                    max_length=transformer_config['max_length']
                )
        
        elif architecture == 'lstm':
            print("Creating LSTM model...")
            self.model = LSTMClassifier(
                num_classes=num_classes,
                vocab_size=10000,
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout'],
                max_length=transformer_config['max_length']
            )
        
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Compile model
        self.model.compile(learning_rate=model_config['learning_rate'])
        print("\nModel architecture:")
        self.model.summary()
    
    def prepare_data(self,
                    texts_train: List[str],
                    labels_train: np.ndarray,
                    texts_val: List[str],
                    labels_val: np.ndarray,
                    context_train: Optional[List[str]] = None,
                    context_val: Optional[List[str]] = None) -> Tuple:
        """
        Prepare data for training
        
        Args:
            texts_train: Training texts
            labels_train: Training labels
            texts_val: Validation texts
            labels_val: Validation labels
            context_train: Training context (for hybrid model)
            context_val: Validation context (for hybrid model)
        
        Returns:
            Tuple of prepared training and validation data
        """
        model_config = self.config.get_model_config()
        architecture = model_config.get('architecture', 'transformer')
        
        if architecture == 'transformer':
            print("Encoding texts with sentence transformer...")
            X_train = self.model.encode_texts(texts_train)
            X_val = self.model.encode_texts(texts_val)
            
            if isinstance(self.model, HybridClassifier) and context_train:
                # Encode context for hybrid model
                context_encoder = self.preprocessor.label_encoder_3
                ctx_train_encoded = context_encoder.transform(context_train)
                ctx_val_encoded = context_encoder.transform(context_val)
                
                return ([X_train, ctx_train_encoded], labels_train,
                       [X_val, ctx_val_encoded], labels_val)
            else:
                return X_train, labels_train, X_val, labels_val
        
        elif architecture == 'lstm':
            print("Creating tokenizer and encoding texts...")
            self.model.create_tokenizer(texts_train + texts_val)
            X_train = self.model.texts_to_sequences(texts_train)
            X_val = self.model.texts_to_sequences(texts_val)
            
            return X_train, labels_train, X_val, labels_val
    
    def train(self,
             X_train,
             y_train,
             X_val,
             y_val,
             class_weights: Optional[Dict[int, float]] = None) -> keras.callbacks.History:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            class_weights: Optional class weights for imbalanced data
        
        Returns:
            Training history
        """
        model_config = self.config.get_model_config()
        training_config = self.config.get_training_config()
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        print(f"\nStarting training for Classification {self.classification_level}...")
        print(f"Training samples: {len(y_train)}")
        print(f"Validation samples: {len(y_val)}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        # Train model
        self.history = self.model.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return self.history
    
    def _setup_callbacks(self) -> List[keras.callbacks.Callback]:
        """Setup training callbacks"""
        model_config = self.config.get_model_config()
        
        callbacks = []
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=model_config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Model checkpoint
        checkpoint_path = self.checkpoints_dir / f"class_{self.classification_level}_best.h5"
        checkpoint = keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=str(self.run_dir),
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def evaluate(self,
                X_test,
                y_test,
                class_names: List[str]) -> Dict:
        """
        Evaluate model and generate reports
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_names: List of class names
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model...")
        
        # Get predictions
        y_pred_proba = self.model.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_acc, _, test_precision, test_recall = self.model.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        # Classification report
        report = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save results
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        # Save to file
        results_path = self.run_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, class_names)
        
        # Plot training history
        self._plot_training_history()
        
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        
        return results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Confusion Matrix - Classification {self.classification_level}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = self.run_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {save_path}")
    
    def _plot_training_history(self):
        """Plot and save training history"""
        if self.history is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        save_path = self.run_dir / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history saved to: {save_path}")
    
    def save_model(self, model_name: Optional[str] = None):
        """Save the trained model"""
        if model_name is None:
            model_name = f"ledger_classifier_level_{self.classification_level}.h5"
        
        model_path = self.model_dir / model_name
        self.model.model.save_weights(str(model_path))
        print(f"Model weights saved to: {model_path}")
        
        # Save model architecture
        arch_path = self.model_dir / f"architecture_level_{self.classification_level}.json"
        with open(arch_path, 'w') as f:
            json.dump(self.model.model.to_json(), f)
        
        # Save training config
        config_path = self.model_dir / f"training_config_level_{self.classification_level}.json"
        with open(config_path, 'w') as f:
            json.dump({
                'classification_level': self.classification_level,
                'timestamp': self.timestamp,
                'model_config': self.config.get_model_config(),
                'training_config': self.config.get_training_config()
            }, f, indent=2)


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
