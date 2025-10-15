import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from config import config, logger
from database import EssayDatabase

class EssayScoringModel(nn.Module):
    """Custom transformer-based model for essay scoring."""
    
    def __init__(self, model_name: str, num_labels: int = 1, dropout_rate: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.config.output_hidden_states = True
        
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-layer scoring head
        self.scoring_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_labels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for the scoring head."""
        for module in self.scoring_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        logits = self.scoring_head(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.float())
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

class EssayScoringTrainer:
    """Trainer class for essay scoring model."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.trainer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def prepare_dataset(self, essays: List[Dict[str, Any]], test_size: float = 0.2) -> Tuple:
        """Prepare dataset for training."""
        texts = [essay['content'] for essay in essays]
        scores = [essay['score'] for essay in essays]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, scores, test_size=test_size, random_state=42
        )
        
        # Tokenize texts
        train_encodings = self.tokenizer(
            X_train, 
            truncation=True, 
            padding=True, 
            max_length=config.max_length,
            return_tensors="pt"
        )
        
        test_encodings = self.tokenizer(
            X_test, 
            truncation=True, 
            padding=True, 
            max_length=config.max_length,
            return_tensors="pt"
        )
        
        # Create datasets
        train_dataset = EssayDataset(train_encodings, y_train)
        test_dataset = EssayDataset(test_encodings, y_test)
        
        return train_dataset, test_dataset, X_test, y_test
    
    def train(self, train_dataset, eval_dataset=None):
        """Train the model."""
        # Initialize model
        self.model = EssayScoringModel(self.model_name)
        self.model.to(self.device)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(config.models_dir),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=str(config.logs_dir),
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=50 if eval_dataset else None,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False if eval_dataset else None,
            learning_rate=config.learning_rate,
            report_to=None,  # Disable wandb
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None,
        )
        
        # Train model
        logger.info("Starting model training...")
        self.trainer.train()
        
        # Save model
        self.save_model()
        logger.info("Model training completed and saved")
    
    def evaluate(self, test_dataset, X_test, y_test):
        """Evaluate the model."""
        if not self.trainer:
            raise ValueError("Model must be trained first")
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        predicted_scores = predictions.predictions.squeeze()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predicted_scores)
        mae = mean_absolute_error(y_test, predicted_scores)
        r2 = r2_score(y_test, predicted_scores)
        
        # Calculate additional metrics
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - predicted_scores) / y_test)) * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Create detailed results
        results = []
        for i, (text, actual, predicted) in enumerate(zip(X_test, y_test, predicted_scores)):
            results.append({
                'index': i,
                'text_preview': text[:100] + "..." if len(text) > 100 else text,
                'actual_score': actual,
                'predicted_score': predicted,
                'error': abs(actual - predicted),
                'error_percentage': abs((actual - predicted) / actual) * 100
            })
        
        return metrics, results
    
    def predict(self, texts: List[str]) -> List[float]:
        """Predict scores for new essays."""
        if not self.model:
            self.load_model()
        
        self.model.eval()
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=config.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.squeeze().cpu().numpy()
        
        # Ensure scores are within valid range
        predictions = np.clip(predictions, config.min_score, config.max_score)
        
        return predictions.tolist()
    
    def save_model(self):
        """Save the trained model."""
        if not self.model:
            raise ValueError("No model to save")
        
        model_path = config.models_dir / "essay_scoring_model"
        model_path.mkdir(exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))
        
        # Save configuration
        config_dict = {
            'model_name': self.model_name,
            'max_length': config.max_length,
            'min_score': config.min_score,
            'max_score': config.max_score
        }
        
        with open(model_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load a trained model."""
        model_path = config.models_dir / "essay_scoring_model"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load configuration
        with open(model_path / "config.json", "r") as f:
            config_dict = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        # Load model
        self.model = EssayScoringModel(config_dict['model_name'])
        self.model.load_state_dict(torch.load(model_path / "pytorch_model.bin", map_location=self.device))
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {model_path}")

class EssayDataset(torch.utils.data.Dataset):
    """Custom dataset for essay scoring."""
    
    def __init__(self, encodings, scores):
        self.encodings = encodings
        self.scores = scores
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.scores[idx], dtype=torch.float)
        return item
    
    def __len__(self):
        return len(self.scores)

def train_and_evaluate_model():
    """Train and evaluate the essay scoring model."""
    # Initialize database and get essays
    db = EssayDatabase()
    essays = db.get_all_essays()
    
    if len(essays) < 5:
        logger.warning("Not enough essays for training. Please add more essays to the database.")
        return None
    
    logger.info(f"Training model with {len(essays)} essays")
    
    # Initialize trainer
    trainer = EssayScoringTrainer()
    
    # Prepare dataset
    train_dataset, test_dataset, X_test, y_test = trainer.prepare_dataset(essays)
    
    # Train model
    trainer.train(train_dataset, test_dataset)
    
    # Evaluate model
    metrics, results = trainer.evaluate(test_dataset, X_test, y_test)
    
    # Print results
    print("\n🧠 Automated Essay Scoring Model Results:")
    print("=" * 50)
    print(f"📊 Model Performance:")
    print(f"   Mean Squared Error: {metrics['mse']:.3f}")
    print(f"   Mean Absolute Error: {metrics['mae']:.3f}")
    print(f"   Root Mean Squared Error: {metrics['rmse']:.3f}")
    print(f"   R² Score: {metrics['r2']:.3f}")
    print(f"   Mean Absolute Percentage Error: {metrics['mape']:.1f}%")
    
    print(f"\n📝 Sample Predictions:")
    for result in results[:3]:  # Show first 3 results
        print(f"\n   Essay: {result['text_preview']}")
        print(f"   Actual Score: {result['actual_score']:.1f}")
        print(f"   Predicted Score: {result['predicted_score']:.1f}")
        print(f"   Error: {result['error']:.2f} ({result['error_percentage']:.1f}%)")
    
    return trainer, metrics

if __name__ == "__main__":
    # Train and evaluate the model
    trainer, metrics = train_and_evaluate_model()
    
    if trainer:
        # Test prediction on new essays
        test_essays = [
            "Artificial intelligence is transforming the way we work and live. It has the potential to solve complex problems and improve efficiency across various industries.",
            "The importance of environmental conservation cannot be overstated. We must take immediate action to protect our planet for future generations."
        ]
        
        predictions = trainer.predict(test_essays)
        
        print(f"\n🔮 Predictions for New Essays:")
        for essay, score in zip(test_essays, predictions):
            print(f"\n   Essay: {essay[:60]}...")
            print(f"   Predicted Score: {score:.2f}")
