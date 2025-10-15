#!/usr/bin/env python3
"""
Project 198: Modern Automated Essay Scoring System

A comprehensive AI-powered essay scoring system using transformer models,
advanced text preprocessing, and modern web interface.

Features:
- Transformer-based scoring models (DistilBERT, BERT, etc.)
- Comprehensive text analysis and feature extraction
- Modern web interface with FastAPI
- Database management with SQLAlchemy
- Advanced evaluation metrics and visualizations
- Real-time scoring and analytics

Usage:
    python 0198.py                    # Run basic demo
    python 0198.py --web              # Start web interface
    python 0198.py --train            # Train model
    python 0198.py --evaluate         # Run comprehensive evaluation
    python 0198.py --help             # Show help
"""

import argparse
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import config, logger
from database import initialize_database, EssayDatabase
from models import train_and_evaluate_model, EssayScoringTrainer
from evaluation import run_comprehensive_evaluation
from preprocessing import analyze_essay_features

def run_basic_demo():
    """Run the basic essay scoring demo."""
    print("🧠 Automated Essay Scoring System - Basic Demo")
    print("=" * 50)
    
    # Initialize database with mock data
    logger.info("Initializing database with mock data...")
    db = initialize_database()
    
    # Get sample essays
    essays = db.get_all_essays()
    print(f"\n📚 Loaded {len(essays)} essays from database")
    
    # Show sample essays
    print(f"\n📝 Sample Essays:")
    for i, essay in enumerate(essays[:3]):
        print(f"\n   {i+1}. {essay['title']}")
        print(f"      Score: {essay['score']}")
        print(f"      Grade Level: {essay['grade_level']}")
        print(f"      Word Count: {essay['word_count']}")
        print(f"      Preview: {essay['content'][:100]}...")
    
    # Train and evaluate model
    print(f"\n🤖 Training AI Model...")
    trainer, metrics = train_and_evaluate_model()
    
    if trainer:
        # Test with new essays
        test_essays = [
            "Artificial intelligence is revolutionizing the way we work and live. It has the potential to solve complex problems and improve efficiency across various industries.",
            "The importance of environmental conservation cannot be overstated. We must take immediate action to protect our planet for future generations.",
            "Education is the foundation of a prosperous society. It empowers individuals with knowledge and critical thinking skills."
        ]
        
        print(f"\n🔮 Testing with New Essays:")
        predictions = trainer.predict(test_essays)
        
        for i, (essay, score) in enumerate(zip(test_essays, predictions)):
            print(f"\n   Essay {i+1}: {essay[:60]}...")
            print(f"   Predicted Score: {score:.2f}")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📊 Model Performance: R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:.3f}")
    else:
        print(f"\n❌ Demo failed. Please check the logs for details.")

def start_web_interface():
    """Start the web interface."""
    print("🌐 Starting Web Interface...")
    print("=" * 30)
    
    try:
        import uvicorn
        from api import app
        
        # Initialize database
        initialize_database()
        
        print(f"🚀 Starting server on http://{config.api_host}:{config.api_port}")
        print(f"📱 Open your browser and navigate to the URL above")
        print(f"🛑 Press Ctrl+C to stop the server")
        
        uvicorn.run(
            app,
            host=config.api_host,
            port=config.api_port,
            reload=config.debug,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print(f"💡 Please install required dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")

def train_model():
    """Train the essay scoring model."""
    print("🤖 Training Essay Scoring Model...")
    print("=" * 40)
    
    # Initialize database
    db = initialize_database()
    essays = db.get_all_essays()
    
    if len(essays) < 5:
        print(f"❌ Not enough essays for training. Need at least 5, found {len(essays)}")
        return
    
    # Train model
    trainer, metrics = train_and_evaluate_model()
    
    if trainer:
        print(f"\n✅ Model training completed successfully!")
        print(f"📊 Performance Metrics:")
        print(f"   R² Score: {metrics['r2']:.3f}")
        print(f"   Mean Absolute Error: {metrics['mae']:.3f}")
        print(f"   Root Mean Squared Error: {metrics['rmse']:.3f}")
        print(f"   Mean Absolute Percentage Error: {metrics['mape']:.1f}%")
    else:
        print(f"❌ Model training failed. Please check the logs for details.")

def run_evaluation():
    """Run comprehensive model evaluation."""
    print("📊 Running Comprehensive Model Evaluation...")
    print("=" * 50)
    
    report = run_comprehensive_evaluation()
    
    if report:
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Report saved to logs directory")
    else:
        print(f"❌ Evaluation failed. Please check the logs for details.")

def analyze_features():
    """Analyze essay features."""
    print("🔍 Analyzing Essay Features...")
    print("=" * 35)
    
    # Initialize database
    db = initialize_database()
    essays = db.get_all_essays()
    
    if not essays:
        print(f"❌ No essays found in database")
        return
    
    # Analyze features
    features_df = analyze_essay_features(essays)
    
    print(f"📊 Feature Analysis Results:")
    print(f"   Total essays analyzed: {len(essays)}")
    print(f"   Total features extracted: {len(features_df.columns) - 4}")
    
    # Show feature statistics
    numeric_features = features_df.select_dtypes(include=['number']).columns
    numeric_features = [col for col in numeric_features if col not in ['essay_id', 'score']]
    
    print(f"\n🔍 Top Features by Correlation with Score:")
    correlations = {}
    for feature in numeric_features:
        try:
            corr = features_df[feature].corr(features_df['score'])
            if not pd.isna(corr):
                correlations[feature] = abs(corr)
        except:
            continue
    
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, corr) in enumerate(sorted_features[:10]):
        print(f"   {i+1:2d}. {feature}: {corr:.3f}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Modern Automated Essay Scoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--web', 
        action='store_true', 
        help='Start web interface'
    )
    parser.add_argument(
        '--train', 
        action='store_true', 
        help='Train the model'
    )
    parser.add_argument(
        '--evaluate', 
        action='store_true', 
        help='Run comprehensive evaluation'
    )
    parser.add_argument(
        '--features', 
        action='store_true', 
        help='Analyze essay features'
    )
    parser.add_argument(
        '--demo', 
        action='store_true', 
        help='Run basic demo (default)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.web:
            start_web_interface()
        elif args.train:
            train_model()
        elif args.evaluate:
            run_evaluation()
        elif args.features:
            analyze_features()
        else:
            # Default: run basic demo
            run_basic_demo()
            
    except KeyboardInterrupt:
        print(f"\n\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()