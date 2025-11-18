"""
Configuration Loader Module
Handles loading and parsing of configuration files
"""

import yaml
import os
from typing import Dict, List, Any
from pathlib import Path


class ConfigLoader:
    """Loads and manages configuration settings"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        paths = self.config.get('paths', {})
        for path_key, path_value in paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def get_classification_hierarchy(self) -> Dict[str, Any]:
        """Get classification hierarchy configuration"""
        return self.config.get('classification_hierarchy', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})
    
    def get_excel_config(self) -> Dict[str, Any]:
        """Get Excel integration configuration"""
        return self.config.get('excel', {})
    
    def get_transformer_config(self) -> Dict[str, Any]:
        """Get transformer-specific configuration"""
        return self.config.get('transformer', {})
    
    def get_classes_for_classification(self, 
                                      classification_1: str,
                                      classification_level: int,
                                      classification_3: str = None) -> List[str]:
        """
        Get available classes for a specific classification level
        
        Args:
            classification_1: Level 1 classification (e.g., 'P&L Account')
            classification_level: Target classification level (3 or 4)
            classification_3: Required for level 4 classification
        
        Returns:
            List of available classes
        """
        hierarchy = self.get_classification_hierarchy()
        
        if classification_1 not in hierarchy:
            raise ValueError(f"Unknown classification_1: {classification_1}")
        
        if classification_level == 3:
            return hierarchy[classification_1].get('classification_3', [])
        
        elif classification_level == 4:
            if not classification_3:
                raise ValueError("classification_3 is required for level 4 classification")
            
            class_4_dict = hierarchy[classification_1].get('classification_4', {})
            return class_4_dict.get(classification_3, [])
        
        else:
            raise ValueError(f"Invalid classification level: {classification_level}")
    
    def get_all_classes(self, classification_level: int) -> List[str]:
        """
        Get all possible classes for a classification level
        
        Args:
            classification_level: Target classification level (3 or 4)
        
        Returns:
            List of all possible classes
        """
        hierarchy = self.get_classification_hierarchy()
        all_classes = []
        
        for class_1, structure in hierarchy.items():
            if classification_level == 3:
                all_classes.extend(structure.get('classification_3', []))
            elif classification_level == 4:
                class_4_dict = structure.get('classification_4', {})
                for class_3, class_4_list in class_4_dict.items():
                    all_classes.extend(class_4_list)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_classes = []
        for cls in all_classes:
            if cls not in seen:
                seen.add(cls)
                unique_classes.append(cls)
        
        return unique_classes


if __name__ == "__main__":
    # Test configuration loader
    config = ConfigLoader()
    print("Configuration loaded successfully!")
    print(f"\nClassification 3 classes for P&L Account:")
    print(config.get_classes_for_classification('P&L Account', 3))
    print(f"\nClassification 4 classes for Cost of Goods Sold:")
    print(config.get_classes_for_classification('P&L Account', 4, 'Cost of Goods Sold'))
