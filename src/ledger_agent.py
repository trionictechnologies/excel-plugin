"""
Ledger Classification Agent
Uses OpenAI API to classify ledgers like a Chartered Accountant
"""

import os
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv
import json

from config_loader import ConfigLoader


class LedgerClassificationAgent:
    """
    AI Agent that classifies ledgers using OpenAI API
    Thinks and reasons like a professional Chartered Accountant
    """
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize the agent
        
        Args:
            config_loader: ConfigLoader instance with classification hierarchy
        """
        load_dotenv()
        
        self.config = config_loader
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in .env file"
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.hierarchy = config_loader.get_classification_hierarchy()
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt that defines the agent's role"""
        
        prompt = """You are an expert Chartered Accountant specializing in ledger classification.
Your task is to classify ledger account names into the correct categories based on accounting principles.

You have deep knowledge of:
- Financial accounting standards (GAAP, IFRS, IndAS)
- Chart of accounts structure
- P&L and Balance Sheet classifications
- Cost accounting principles

You think systematically and provide accurate classifications with reasoning.

IMPORTANT: You must ONLY return valid JSON in the exact format specified. No additional text."""
        
        return prompt
    
    def _build_classification_context(self, 
                                     classification_level: int,
                                     classification_1: str = "P&L Account",
                                     classification_3: Optional[str] = None) -> str:
        """Build context with available categories"""
        
        if classification_level == 3:
            categories = self.config.get_classes_for_classification(
                classification_1, 3
            )
            context = f"""
Available Classification 3 categories:
{chr(10).join(f"- {cat}" for cat in categories)}

Your task: Classify the ledger into ONE of these categories.
"""
        
        elif classification_level == 4:
            if not classification_3:
                raise ValueError("classification_3 required for level 4")
            
            categories = self.config.get_classes_for_classification(
                classification_1, 4, classification_3
            )
            context = f"""
The ledger is classified as: {classification_3}

Available Classification 4 sub-categories for "{classification_3}":
{chr(10).join(f"- {cat}" for cat in categories)}

Your task: Classify the ledger into ONE of these sub-categories.
"""
        
        else:
            raise ValueError(f"Invalid classification level: {classification_level}")
        
        return context
    
    def classify_single(self,
                       ledger_name: str,
                       classification_level: int = 3,
                       classification_1: str = "P&L Account",
                       classification_3: Optional[str] = None) -> Tuple[str, float, str]:
        """
        Classify a single ledger
        
        Args:
            ledger_name: Name of the ledger to classify
            classification_level: 3 or 4
            classification_1: Level 1 category (default: P&L Account)
            classification_3: Required for level 4
        
        Returns:
            Tuple of (classification, confidence, reasoning)
        """
        
        # Build the classification request
        context = self._build_classification_context(
            classification_level, classification_1, classification_3
        )
        
        user_prompt = f"""{context}

Ledger Name: "{ledger_name}"

Analyze this ledger name and classify it. Return your response as JSON:

{{
    "classification": "The exact category name from the list",
    "confidence": 0.95,
    "reasoning": "Brief explanation of why you chose this category"
}}

Confidence should be between 0 and 1 (e.g., 0.95 for 95% confident).
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return (
                result['classification'],
                float(result['confidence']),
                result['reasoning']
            )
        
        except Exception as e:
            print(f"Error classifying '{ledger_name}': {e}")
            return ("Error", 0.0, str(e))
    
    def classify_batch(self,
                      ledger_names: List[str],
                      classification_level: int = 3,
                      classification_1: str = "P&L Account",
                      classification_3_list: Optional[List[str]] = None) -> List[Dict]:
        """
        Classify multiple ledgers
        
        Args:
            ledger_names: List of ledger names
            classification_level: 3 or 4
            classification_1: Level 1 category
            classification_3_list: List of classification_3 values (for level 4)
        
        Returns:
            List of classification results
        """
        results = []
        
        for i, ledger_name in enumerate(ledger_names):
            classification_3 = None
            if classification_3_list and i < len(classification_3_list):
                classification_3 = classification_3_list[i]
            
            classification, confidence, reasoning = self.classify_single(
                ledger_name,
                classification_level,
                classification_1,
                classification_3
            )
            
            results.append({
                'ledger_name': ledger_name,
                'classification': classification,
                'confidence': confidence,
                'reasoning': reasoning
            })
        
        return results
    
    def classify_with_alternatives(self,
                                  ledger_name: str,
                                  classification_level: int = 3,
                                  classification_1: str = "P&L Account",
                                  classification_3: Optional[str] = None,
                                  top_k: int = 3) -> List[Dict]:
        """
        Get top K classification suggestions with reasoning
        
        Args:
            ledger_name: Ledger name to classify
            classification_level: 3 or 4
            classification_1: Level 1 category
            classification_3: Required for level 4
            top_k: Number of alternatives to return
        
        Returns:
            List of top K classifications with confidence and reasoning
        """
        
        context = self._build_classification_context(
            classification_level, classification_1, classification_3
        )
        
        user_prompt = f"""{context}

Ledger Name: "{ledger_name}"

Provide the top {top_k} most likely classifications with confidence scores and reasoning.

Return JSON array:
[
    {{
        "classification": "Category name",
        "confidence": 0.85,
        "reasoning": "Why this is likely"
    }},
    ...
]

Order by confidence (highest first). Confidences should sum to approximately 1.0.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Handle both array and object with "classifications" key
            if isinstance(result, list):
                return result
            elif 'classifications' in result:
                return result['classifications']
            elif 'alternatives' in result:
                return result['alternatives']
            else:
                # If single result, wrap in array
                return [result]
        
        except Exception as e:
            print(f"Error: {e}")
            return [{"classification": "Error", "confidence": 0.0, "reasoning": str(e)}]


if __name__ == "__main__":
    # Test the agent
    config = ConfigLoader('../config.yaml')
    agent = LedgerClassificationAgent(config)
    
    # Test single classification
    test_ledger = "Purchase of Raw Material"
    classification, confidence, reasoning = agent.classify_single(test_ledger, 3)
    
    print(f"Ledger: {test_ledger}")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Reasoning: {reasoning}")
