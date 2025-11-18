"""
Inference Engine Module
Handles real-time classification of ledgers
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import joblib
import json

from config_loader import ConfigLoader
from data_preprocessor import DataPreprocessor
from models import TransformerClassifier, LSTMClassifier, HybridClassifier


class LedgerClassifier:
    """Real-time ledger classification engine"""
    
    def __init__(self, config_loader: ConfigLoader, classification_level: int = 3):
        """
        Initialize classifier
        
        Args:
            config_loader: ConfigLoader instance
            classification_level: Target classification level (3 or 4)
        """
        self.config = config_loader
        self.classification_level = classification_level
        self.preprocessor = DataPreprocessor(config_loader)
        self.model = None
        self.is_loaded = False
        
        self.model_dir = Path(self.config.config['paths']['model_dir'])
        self.confidence_threshold = self.config.get_excel_config().get('confidence_threshold', 0.7)
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load trained model
        
        Args:
            model_path: Optional custom model path
        """
        if model_path is None:
            model_path = self.model_dir / f"ledger_classifier_level_{self.classification_level}.h5"
        
        # Load preprocessor encoders
        self.preprocessor.load_encoders(str(self.model_dir))
        
        # Get class names
        if self.classification_level == 3:
            class_names = self.preprocessor.get_class_names(3)
        else:
            class_names = self.preprocessor.get_class_names(4)
        
        if not class_names:
            raise ValueError(f"No classes found for classification level {self.classification_level}")
        
        num_classes = len(class_names)
        
        # Load model configuration
        config_path = self.model_dir / f"training_config_level_{self.classification_level}.json"
        with open(config_path, 'r') as f:
            training_config = json.load(f)
        
        model_config = training_config['model_config']
        architecture = model_config.get('architecture', 'transformer')
        
        # Create model
        if architecture == 'transformer':
            if self.classification_level == 4:
                num_context_classes = len(self.preprocessor.get_class_names(3))
                self.model = HybridClassifier(
                    num_classes=num_classes,
                    num_context_classes=num_context_classes,
                    pretrained_model=self.config.get_transformer_config()['pretrained_model'],
                    hidden_dim=model_config['hidden_dim'],
                    dropout=model_config['dropout']
                )
            else:
                self.model = TransformerClassifier(
                    num_classes=num_classes,
                    pretrained_model=self.config.get_transformer_config()['pretrained_model'],
                    hidden_dim=model_config['hidden_dim'],
                    dropout=model_config['dropout']
                )
        elif architecture == 'lstm':
            self.model = LSTMClassifier(
                num_classes=num_classes,
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout']
            )
            # Load tokenizer for LSTM
            tokenizer_path = self.model_dir / f"tokenizer_level_{self.classification_level}.pkl"
            self.model.tokenizer = joblib.load(str(tokenizer_path))
        
        # Load weights
        self.model.model.load_weights(str(model_path))
        self.is_loaded = True
        
        print(f"Model loaded successfully for classification level {self.classification_level}")
        print(f"Number of classes: {num_classes}")
    
    def classify_single(self,
                       ledger_name: str,
                       classification_3: Optional[str] = None) -> Tuple[str, float]:
        """
        Classify a single ledger
        
        Args:
            ledger_name: Ledger name to classify
            classification_3: Required for level 4 classification
        
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Validate inputs
        if self.classification_level == 4 and not classification_3:
            raise ValueError("classification_3 required for level 4 classification")
        
        # Clean text
        cleaned_text = self.preprocessor.clean_ledger_name(ledger_name)
        
        # Prepare input
        model_config = self.config.get_model_config()
        architecture = model_config.get('architecture', 'transformer')
        
        if architecture == 'transformer':
            text_embedding = self.model.encode_texts([cleaned_text])
            
            if isinstance(self.model, HybridClassifier):
                # Encode context
                context_encoder = self.preprocessor.label_encoder_3
                context_encoded = context_encoder.transform([classification_3])
                X = [text_embedding, context_encoded.reshape(-1, 1)]
            else:
                X = text_embedding
        
        elif architecture == 'lstm':
            X = self.model.texts_to_sequences([cleaned_text])
        
        # Predict
        predictions = self.model.model.predict(X, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        
        # Get class name
        if self.classification_level == 3:
            predicted_class = self.preprocessor.label_encoder_3.inverse_transform([predicted_idx])[0]
        else:
            predicted_class = self.preprocessor.label_encoder_4.inverse_transform([predicted_idx])[0]
        
        return predicted_class, confidence
    
    def classify_batch(self,
                      ledger_names: List[str],
                      classification_3_list: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Classify multiple ledgers
        
        Args:
            ledger_names: List of ledger names
            classification_3_list: List of classification_3 values (for level 4)
        
        Returns:
            List of (predicted_class, confidence) tuples
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Validate inputs
        if self.classification_level == 4:
            if not classification_3_list or len(classification_3_list) != len(ledger_names):
                raise ValueError("classification_3_list must match ledger_names length for level 4")
        
        # Clean texts
        cleaned_texts = [self.preprocessor.clean_ledger_name(name) for name in ledger_names]
        
        # Prepare inputs
        model_config = self.config.get_model_config()
        architecture = model_config.get('architecture', 'transformer')
        
        if architecture == 'transformer':
            text_embeddings = self.model.encode_texts(cleaned_texts)
            
            if isinstance(self.model, HybridClassifier):
                context_encoder = self.preprocessor.label_encoder_3
                context_encoded = context_encoder.transform(classification_3_list)
                X = [text_embeddings, context_encoded.reshape(-1, 1)]
            else:
                X = text_embeddings
        
        elif architecture == 'lstm':
            X = self.model.texts_to_sequences(cleaned_texts)
        
        # Predict
        predictions = self.model.model.predict(X, verbose=0)
        predicted_indices = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        # Get class names
        if self.classification_level == 3:
            predicted_classes = self.preprocessor.label_encoder_3.inverse_transform(predicted_indices)
        else:
            predicted_classes = self.preprocessor.label_encoder_4.inverse_transform(predicted_indices)
        
        results = list(zip(predicted_classes, confidences.tolist()))
        return results
    
    def classify_dataframe(self,
                          df: pd.DataFrame,
                          input_column: str = 'Ledger Name',
                          classification_3_column: Optional[str] = None) -> pd.DataFrame:
        """
        Classify ledgers in a DataFrame
        
        Args:
            df: Input DataFrame
            input_column: Column containing ledger names
            classification_3_column: Column with classification 3 (for level 4)
        
        Returns:
            DataFrame with classifications added
        """
        if input_column not in df.columns:
            raise ValueError(f"Column '{input_column}' not found in DataFrame")
        
        if self.classification_level == 4 and classification_3_column not in df.columns:
            raise ValueError(f"Column '{classification_3_column}' required for level 4")
        
        # Get data to classify
        ledger_names = df[input_column].tolist()
        
        if self.classification_level == 4:
            classification_3_list = df[classification_3_column].tolist()
        else:
            classification_3_list = None
        
        # Classify
        results = self.classify_batch(ledger_names, classification_3_list)
        
        # Add results to DataFrame
        output_column = f'Classification {self.classification_level}'
        confidence_column = f'{output_column} Confidence'
        
        df[output_column] = [r[0] for r in results]
        df[confidence_column] = [r[1] for r in results]
        
        # Flag low confidence predictions
        df[f'{output_column} Review'] = df[confidence_column] < self.confidence_threshold
        
        return df
    
    def get_top_predictions(self,
                           ledger_name: str,
                           top_k: int = 3,
                           classification_3: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Get top K predictions with confidences
        
        Args:
            ledger_name: Ledger name to classify
            top_k: Number of top predictions to return
            classification_3: Required for level 4
        
        Returns:
            List of (class_name, confidence) tuples
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Clean text
        cleaned_text = self.preprocessor.clean_ledger_name(ledger_name)
        
        # Prepare input
        model_config = self.config.get_model_config()
        architecture = model_config.get('architecture', 'transformer')
        
        if architecture == 'transformer':
            text_embedding = self.model.encode_texts([cleaned_text])
            
            if isinstance(self.model, HybridClassifier):
                context_encoder = self.preprocessor.label_encoder_3
                context_encoded = context_encoder.transform([classification_3])
                X = [text_embedding, context_encoded.reshape(-1, 1)]
            else:
                X = text_embedding
        
        elif architecture == 'lstm':
            X = self.model.texts_to_sequences([cleaned_text])
        
        # Predict
        predictions = self.model.model.predict(X, verbose=0)[0]
        
        # Get top K
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        # Get class names
        if self.classification_level == 3:
            encoder = self.preprocessor.label_encoder_3
        else:
            encoder = self.preprocessor.label_encoder_4
        
        top_classes = encoder.inverse_transform(top_indices)
        top_confidences = predictions[top_indices]
        
        return list(zip(top_classes, top_confidences.tolist()))


if __name__ == "__main__":
    from config_loader import ConfigLoader
    
    config = ConfigLoader()
    
    # Test classifier initialization
    classifier = LedgerClassifier(config, classification_level=3)
    print("Classifier initialized successfully!")
