"""
Data Preprocessing Module
Handles data loading, cleaning, and preparation for model training
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import joblib
from pathlib import Path


class DataPreprocessor:
    """Preprocesses ledger data for classification"""
    
    def __init__(self, config_loader):
        """
        Initialize data preprocessor
        
        Args:
            config_loader: ConfigLoader instance
        """
        self.config = config_loader
        self.label_encoder_3 = LabelEncoder()
        self.label_encoder_4 = LabelEncoder()
        self.class_3_fitted = False
        self.class_4_fitted = False
    
    def clean_ledger_name(self, text: str) -> str:
        """
        Clean and normalize ledger name
        
        Args:
            text: Raw ledger name
        
        Returns:
            Cleaned ledger name
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def augment_text(self, text: str) -> List[str]:
        """
        Generate augmented versions of text for training
        
        Args:
            text: Original text
        
        Returns:
            List of augmented texts
        """
        augmented = [text]
        
        # Add variations
        # Remove common prefixes/suffixes
        variations = [
            text.replace('account', '').strip(),
            text.replace('expense', '').strip(),
            text.replace('expenses', '').strip(),
            text.replace('a/c', '').strip(),
        ]
        
        augmented.extend([v for v in variations if v and v != text])
        
        return augmented
    
    def load_training_data(self, 
                          file_path: str,
                          classification_level: int = 3) -> pd.DataFrame:
        """
        Load training data from Excel/CSV file
        
        Args:
            file_path: Path to training data file
            classification_level: Target classification level (3 or 4)
        
        Returns:
            DataFrame with training data
        """
        # Determine file type and load accordingly
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Validate required columns
        required_columns = ['Ledger Name', f'Classification {classification_level}']
        if classification_level == 4:
            required_columns.append('Classification 3')
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    def prepare_training_data(self,
                             df: pd.DataFrame,
                             classification_level: int = 3,
                             augment: bool = True) -> Tuple[List[str], np.ndarray, Optional[List[str]]]:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            classification_level: Target classification level (3 or 4)
            augment: Whether to apply data augmentation
        
        Returns:
            Tuple of (texts, labels, context) where context is classification_3 for level 4
        """
        # Drop rows with missing values
        required_cols = ['Ledger Name', f'Classification {classification_level}']
        if classification_level == 4:
            required_cols.append('Classification 3')
        
        df = df.dropna(subset=required_cols)
        
        texts = []
        labels = []
        context = [] if classification_level == 4 else None
        
        for _, row in df.iterrows():
            ledger_name = self.clean_ledger_name(row['Ledger Name'])
            label = row[f'Classification {classification_level}']
            
            if augment:
                augmented_texts = self.augment_text(ledger_name)
            else:
                augmented_texts = [ledger_name]
            
            for aug_text in augmented_texts:
                texts.append(aug_text)
                labels.append(label)
                
                if classification_level == 4:
                    context.append(row['Classification 3'])
        
        # Encode labels
        if classification_level == 3:
            if not self.class_3_fitted:
                labels_encoded = self.label_encoder_3.fit_transform(labels)
                self.class_3_fitted = True
            else:
                labels_encoded = self.label_encoder_3.transform(labels)
        else:
            if not self.class_4_fitted:
                labels_encoded = self.label_encoder_4.fit_transform(labels)
                self.class_4_fitted = True
            else:
                labels_encoded = self.label_encoder_4.transform(labels)
        
        return texts, labels_encoded, context
    
    def split_data(self,
                   texts: List[str],
                   labels: np.ndarray,
                   context: Optional[List[str]] = None,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple:
        """
        Split data into training and validation sets
        
        Args:
            texts: List of text samples
            labels: Encoded labels
            context: Optional context (classification_3 for level 4)
            test_size: Proportion of test set
            random_state: Random seed
        
        Returns:
            Tuple of (X_train, X_val, y_train, y_val, context_train, context_val)
        """
        if context is not None:
            X_train, X_val, y_train, y_val, ctx_train, ctx_val = train_test_split(
                texts, labels, context,
                test_size=test_size,
                random_state=random_state,
                stratify=labels
            )
            return X_train, X_val, y_train, y_val, ctx_train, ctx_val
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                texts, labels,
                test_size=test_size,
                random_state=random_state,
                stratify=labels
            )
            return X_train, X_val, y_train, y_val, None, None
    
    def save_encoders(self, save_dir: str = "models"):
        """Save label encoders to disk"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        if self.class_3_fitted:
            joblib.dump(self.label_encoder_3, f"{save_dir}/label_encoder_3.pkl")
        
        if self.class_4_fitted:
            joblib.dump(self.label_encoder_4, f"{save_dir}/label_encoder_4.pkl")
    
    def load_encoders(self, load_dir: str = "models"):
        """Load label encoders from disk"""
        try:
            self.label_encoder_3 = joblib.load(f"{load_dir}/label_encoder_3.pkl")
            self.class_3_fitted = True
        except FileNotFoundError:
            print("Warning: Classification 3 encoder not found")
        
        try:
            self.label_encoder_4 = joblib.load(f"{load_dir}/label_encoder_4.pkl")
            self.class_4_fitted = True
        except FileNotFoundError:
            print("Warning: Classification 4 encoder not found")
    
    def get_class_names(self, classification_level: int) -> List[str]:
        """Get class names for a classification level"""
        if classification_level == 3 and self.class_3_fitted:
            return list(self.label_encoder_3.classes_)
        elif classification_level == 4 and self.class_4_fitted:
            return list(self.label_encoder_4.classes_)
        else:
            return []


if __name__ == "__main__":
    from config_loader import ConfigLoader
    
    config = ConfigLoader()
    preprocessor = DataPreprocessor(config)
    
    # Test text cleaning
    test_text = "Purchase of Raw Material A/c (2023)"
    cleaned = preprocessor.clean_ledger_name(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    print(f"Augmented: {preprocessor.augment_text(cleaned)}")
