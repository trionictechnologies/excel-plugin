"""
Deep Learning Models for Ledger Classification
Implements Transformer and LSTM-based architectures
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import torch


class TransformerClassifier:
    """
    Transformer-based text classifier using sentence transformers
    """
    
    def __init__(self, 
                 num_classes: int,
                 pretrained_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 hidden_dim: int = 256,
                 dropout: float = 0.3,
                 max_length: int = 64):
        """
        Initialize transformer classifier
        
        Args:
            num_classes: Number of output classes
            pretrained_model: Pretrained sentence transformer model
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            max_length: Maximum sequence length
        """
        self.num_classes = num_classes
        self.pretrained_model_name = pretrained_model
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.max_length = max_length
        
        # Load pretrained sentence transformer
        self.sentence_encoder = SentenceTransformer(pretrained_model)
        self.embedding_dim = self.sentence_encoder.get_sentence_embedding_dimension()
        
        # Build classification head
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build the classification model"""
        # Input layer for sentence embeddings
        inputs = layers.Input(shape=(self.embedding_dim,), name='embeddings')
        
        # Dense layers with dropout
        x = layers.Dense(self.hidden_dim, activation='relu')(inputs)
        x = layers.Dropout(self.dropout)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(self.hidden_dim // 2, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(self.hidden_dim // 4, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='transformer_classifier')
        return model
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using sentence transformer
        
        Args:
            texts: List of text strings
        
        Returns:
            Array of sentence embeddings
        """
        embeddings = self.sentence_encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings
    
    def compile(self, learning_rate: float = 0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.SparseCategoricalAccuracy(name='sparse_accuracy'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
    
    def summary(self):
        """Print model summary"""
        self.model.summary()


class LSTMClassifier:
    """
    LSTM-based text classifier with word embeddings
    """
    
    def __init__(self,
                 num_classes: int,
                 vocab_size: int = 10000,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 max_length: int = 64):
        """
        Initialize LSTM classifier
        
        Args:
            num_classes: Number of output classes
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            max_length: Maximum sequence length
        """
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_length = max_length
        
        # Tokenizer will be initialized during training
        self.tokenizer = None
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build the LSTM model"""
        # Input layer
        inputs = layers.Input(shape=(self.max_length,), name='text_input')
        
        # Embedding layer
        x = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            mask_zero=True
        )(inputs)
        
        # LSTM layers
        for i in range(self.num_layers):
            return_sequences = i < self.num_layers - 1
            x = layers.Bidirectional(
                layers.LSTM(
                    self.hidden_dim // 2,
                    return_sequences=return_sequences,
                    dropout=self.dropout
                )
            )(x)
        
        # Dense layers
        x = layers.Dense(self.hidden_dim // 2, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(self.hidden_dim // 4, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='lstm_classifier')
        return model
    
    def create_tokenizer(self, texts: List[str]):
        """
        Create and fit tokenizer on texts
        
        Args:
            texts: List of training texts
        """
        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=self.vocab_size,
            oov_token='<OOV>',
            lower=True
        )
        self.tokenizer.fit_on_texts(texts)
    
    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to padded sequences
        
        Args:
            texts: List of texts
        
        Returns:
            Padded sequences array
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call create_tokenizer first.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )
        return padded
    
    def compile(self, learning_rate: float = 0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.SparseCategoricalAccuracy(name='sparse_accuracy'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
    
    def summary(self):
        """Print model summary"""
        self.model.summary()


class HybridClassifier:
    """
    Hybrid model that combines context (classification_3) with ledger name
    for classification_4 prediction
    """
    
    def __init__(self,
                 num_classes: int,
                 num_context_classes: int,
                 pretrained_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        """
        Initialize hybrid classifier
        
        Args:
            num_classes: Number of output classes (classification_4)
            num_context_classes: Number of context classes (classification_3)
            pretrained_model: Pretrained sentence transformer model
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        self.num_classes = num_classes
        self.num_context_classes = num_context_classes
        self.pretrained_model_name = pretrained_model
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Load sentence transformer
        self.sentence_encoder = SentenceTransformer(pretrained_model)
        self.embedding_dim = self.sentence_encoder.get_sentence_embedding_dimension()
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build the hybrid model with two inputs"""
        # Text embedding input
        text_input = layers.Input(shape=(self.embedding_dim,), name='text_embeddings')
        
        # Context input (classification_3 as one-hot or index)
        context_input = layers.Input(shape=(1,), name='context_input')
        
        # Process text
        text_branch = layers.Dense(self.hidden_dim, activation='relu')(text_input)
        text_branch = layers.Dropout(self.dropout)(text_branch)
        text_branch = layers.BatchNormalization()(text_branch)
        
        # Process context
        context_embedding = layers.Embedding(
            input_dim=self.num_context_classes,
            output_dim=64
        )(context_input)
        context_branch = layers.Flatten()(context_embedding)
        context_branch = layers.Dense(128, activation='relu')(context_branch)
        context_branch = layers.Dropout(self.dropout)(context_branch)
        
        # Concatenate branches
        combined = layers.Concatenate()([text_branch, context_branch])
        
        # Final dense layers
        x = layers.Dense(self.hidden_dim, activation='relu')(combined)
        x = layers.Dropout(self.dropout)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(self.hidden_dim // 2, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = keras.Model(
            inputs=[text_input, context_input],
            outputs=outputs,
            name='hybrid_classifier'
        )
        return model
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence transformer"""
        embeddings = self.sentence_encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings
    
    def compile(self, learning_rate: float = 0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.SparseCategoricalAccuracy(name='sparse_accuracy'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
    
    def summary(self):
        """Print model summary"""
        self.model.summary()


if __name__ == "__main__":
    # Test model creation
    print("Creating Transformer Classifier...")
    transformer = TransformerClassifier(num_classes=13)
    transformer.compile()
    transformer.summary()
    
    print("\n" + "="*50 + "\n")
    print("Creating LSTM Classifier...")
    lstm = LSTMClassifier(num_classes=13)
    lstm.compile()
    lstm.summary()
    
    print("\n" + "="*50 + "\n")
    print("Creating Hybrid Classifier...")
    hybrid = HybridClassifier(num_classes=25, num_context_classes=13)
    hybrid.compile()
    hybrid.summary()
