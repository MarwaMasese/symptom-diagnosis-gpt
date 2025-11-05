"""
Data preparation module for Symptom-Diagnosis-GPT.
Creates or loads symptom-diagnosis datasets and prepares them for training.
"""
import os
import json
import pickle
import random
from typing import List, Tuple, Dict, Any
import tiktoken
import pandas as pd
from pathlib import Path

from .config import get_model_config


class SymptomDatasetBuilder:
    """Build and preprocess symptom-diagnosis dataset."""
    
    def __init__(self, config=None):
        self.config = config or get_model_config()
        self.tokenizer = None
        self.vocab_size = None
        
    def _init_tokenizer(self):
        """Initialize the tokenizer."""
        try:
            # Use GPT-2 tokenizer as base
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.vocab_size = self.tokenizer.n_vocab
            print(f"✅ Tokenizer initialized with vocab size: {self.vocab_size}")
        except Exception as e:
            print(f"❌ Error initializing tokenizer: {e}")
            raise
    
    def create_synthetic_dataset(self, num_samples: int = 1000) -> List[Dict[str, str]]:
        """Create a synthetic symptom-diagnosis dataset."""
        
        # Extended symptom-diagnosis pairs for more realistic training
        symptoms_diagnoses = [
            ("fever, cough, sore throat", "common cold"),
            ("chest pain, shortness of breath", "heart disease"),
            ("headache, nausea, vomiting", "migraine"),
            ("fatigue, joint pain, rash", "lupus"),
            ("stomach pain, diarrhea, fever", "gastroenteritis"),
            ("cough, fever, difficulty breathing", "pneumonia"),
            ("skin rash, itching, swelling", "allergic reaction"),
            ("dizziness, blurred vision, confusion", "hypoglycemia"),
            ("back pain, muscle stiffness, numbness", "herniated disc"),
            ("frequent urination, excessive thirst", "diabetes"),
            ("weight loss, night sweats, persistent cough", "tuberculosis"),
            ("severe headache, neck stiffness, fever", "meningitis"),
            ("rapid heartbeat, sweating, anxiety", "panic disorder"),
            ("yellowing of skin, dark urine, fatigue", "hepatitis"),
            ("persistent cough, wheezing, chest tightness", "asthma"),
            ("memory loss, confusion, difficulty speaking", "alzheimer's disease"),
            ("joint swelling, morning stiffness, pain", "rheumatoid arthritis"),
            ("high fever, severe headache, muscle pain", "malaria"),
            ("persistent sadness, fatigue, loss of interest", "depression"),
            ("tremor, slow movement, muscle rigidity", "parkinson's disease"),
        ]
        
        # Symptom modifiers to create variations
        modifiers = [
            "severe", "mild", "persistent", "occasional", "sudden onset",
            "chronic", "acute", "intermittent", "constant", "worsening"
        ]
        
        additional_symptoms = [
            "loss of appetite", "insomnia", "excessive sweating", "weakness",
            "pale skin", "rapid pulse", "low-grade fever", "dry mouth",
            "muscle aches", "sensitivity to light", "difficulty concentrating"
        ]
        
        dataset = []
        
        for _ in range(num_samples):
            # Choose base symptom-diagnosis pair
            base_symptoms, diagnosis = random.choice(symptoms_diagnoses)
            
            # Add variations
            symptoms_list = base_symptoms.split(", ")
            
            # Sometimes add modifiers
            if random.random() < 0.3:
                modifier = random.choice(modifiers)
                symptoms_list[0] = f"{modifier} {symptoms_list[0]}"
            
            # Sometimes add additional symptoms
            if random.random() < 0.4:
                additional = random.choice(additional_symptoms)
                symptoms_list.append(additional)
            
            # Create the text entry
            symptoms_text = ", ".join(symptoms_list)
            
            # Format as training example
            text = f"Symptoms: {symptoms_text}\nDiagnosis: {diagnosis}"
            
            dataset.append({
                "text": text,
                "symptoms": symptoms_text,
                "diagnosis": diagnosis
            })
        
        return dataset
    
    def load_external_dataset(self, file_path: str) -> List[Dict[str, str]]:
        """Load dataset from external file (CSV, JSON, etc.)."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"⚠️  External dataset not found: {file_path}")
            return []
        
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
                # Assume columns: 'symptoms', 'diagnosis'
                dataset = []
                for _, row in df.iterrows():
                    text = f"Symptoms: {row['symptoms']}\nDiagnosis: {row['diagnosis']}"
                    dataset.append({
                        "text": text,
                        "symptoms": row['symptoms'],
                        "diagnosis": row['diagnosis']
                    })
                return dataset
            
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return data if isinstance(data, list) else [data]
            
        except Exception as e:
            print(f"❌ Error loading external dataset: {e}")
            return []
    
    def tokenize_dataset(self, dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        """Tokenize the dataset for training."""
        if not self.tokenizer:
            self._init_tokenizer()
        
        tokenized_texts = []
        labels = []
        
        for item in dataset:
            # Tokenize the full text
            tokens = self.tokenizer.encode(item["text"])
            
            # Pad or truncate to max_length
            if len(tokens) > self.config.max_length:
                tokens = tokens[:self.config.max_length]
            else:
                # Pad with a special token (using 0 for simplicity)
                tokens = tokens + [0] * (self.config.max_length - len(tokens))
            
            tokenized_texts.append(tokens)
            
            # For simplicity, use the same tokens as labels (next token prediction)
            labels.append(tokens[1:] + [0])  # Shift by one for next token prediction
        
        return {
            "input_ids": tokenized_texts,
            "labels": labels,
            "original_data": dataset
        }
    
    def split_dataset(self, tokenized_data: Dict[str, Any]) -> Tuple[Dict, Dict, Dict]:
        """Split dataset into train, validation, and test sets."""
        total_samples = len(tokenized_data["input_ids"])
        
        # Calculate split indices
        train_end = int(total_samples * self.config.train_split)
        val_end = int(total_samples * (self.config.train_split + self.config.val_split))
        
        # Create indices and shuffle
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        def create_split(indices):
            return {
                "input_ids": [tokenized_data["input_ids"][i] for i in indices],
                "labels": [tokenized_data["labels"][i] for i in indices],
                "original_data": [tokenized_data["original_data"][i] for i in indices]
            }
        
        train_data = create_split(train_indices)
        val_data = create_split(val_indices)
        test_data = create_split(test_indices)
        
        print(f"📊 Dataset split:")
        print(f"  Training samples: {len(train_data['input_ids'])}")
        print(f"  Validation samples: {len(val_data['input_ids'])}")
        print(f"  Test samples: {len(test_data['input_ids'])}")
        
        return train_data, val_data, test_data
    
    def save_processed_data(self, train_data, val_data, test_data, save_dir: str = None):
        """Save processed data to disk."""
        save_dir = save_dir or self.config.processed_data_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Save datasets
        with open(f"{save_dir}/train_data.pkl", "wb") as f:
            pickle.dump(train_data, f)
        
        with open(f"{save_dir}/val_data.pkl", "wb") as f:
            pickle.dump(val_data, f)
        
        with open(f"{save_dir}/test_data.pkl", "wb") as f:
            pickle.dump(test_data, f)
        
        # Save tokenizer info
        tokenizer_info = {
            "vocab_size": self.vocab_size,
            "max_length": self.config.max_length
        }
        
        with open(f"{save_dir}/tokenizer_info.json", "w") as f:
            json.dump(tokenizer_info, f, indent=2)
        
        print(f"✅ Processed data saved to {save_dir}")
    
    def load_processed_data(self, save_dir: str = None) -> Tuple[Dict, Dict, Dict]:
        """Load processed data from disk."""
        save_dir = save_dir or self.config.processed_data_dir
        
        try:
            with open(f"{save_dir}/train_data.pkl", "rb") as f:
                train_data = pickle.load(f)
            
            with open(f"{save_dir}/val_data.pkl", "rb") as f:
                val_data = pickle.load(f)
            
            with open(f"{save_dir}/test_data.pkl", "rb") as f:
                test_data = pickle.load(f)
            
            print(f"✅ Processed data loaded from {save_dir}")
            return train_data, val_data, test_data
            
        except FileNotFoundError:
            print(f"⚠️  Processed data not found in {save_dir}")
            return None, None, None


def build_dataset(num_samples: int = 1000, external_file: str = None):
    """Main function to build and prepare the dataset."""
    print("🔄 Building symptom-diagnosis dataset...")
    
    builder = SymptomDatasetBuilder()
    
    # Create or load dataset
    if external_file and os.path.exists(external_file):
        print(f"📁 Loading external dataset from {external_file}")
        dataset = builder.load_external_dataset(external_file)
        if not dataset:
            print("🔄 Falling back to synthetic dataset...")
            dataset = builder.create_synthetic_dataset(num_samples)
    else:
        print(f"🔄 Creating synthetic dataset with {num_samples} samples...")
        dataset = builder.create_synthetic_dataset(num_samples)
    
    # Tokenize dataset
    print("🔄 Tokenizing dataset...")
    tokenized_data = builder.tokenize_dataset(dataset)
    
    # Split dataset
    print("🔄 Splitting dataset...")
    train_data, val_data, test_data = builder.split_dataset(tokenized_data)
    
    # Save processed data
    print("🔄 Saving processed data...")
    builder.save_processed_data(train_data, val_data, test_data)
    
    print("✅ Dataset preparation complete!")
    return train_data, val_data, test_data


def load_dataset():
    """Load existing processed dataset."""
    builder = SymptomDatasetBuilder()
    return builder.load_processed_data()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare symptom-diagnosis dataset")
    parser.add_argument("--num-samples", type=int, default=1000, 
                       help="Number of synthetic samples to generate")
    parser.add_argument("--external-file", type=str, default=None,
                       help="Path to external dataset file")
    
    args = parser.parse_args()
    
    build_dataset(num_samples=args.num_samples, external_file=args.external_file)
