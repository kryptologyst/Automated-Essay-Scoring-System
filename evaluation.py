import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from config import config, logger
from database import EssayDatabase
from models import EssayScoringTrainer
from preprocessing import analyze_essay_features

class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self):
        self.results = {}
        self.features_df = None
    
    def evaluate_model_performance(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {
            # Basic regression metrics
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'explained_variance': explained_variance_score(y_true, y_pred),
            
            # Additional metrics
            'max_error': np.max(np.abs(y_true - y_pred)),
            'mean_error': np.mean(y_true - y_pred),
            'std_error': np.std(y_true - y_pred),
            
            # Correlation metrics
            'pearson_correlation': stats.pearsonr(y_true, y_pred)[0],
            'spearman_correlation': stats.spearmanr(y_true, y_pred)[0],
            
            # Score distribution metrics
            'score_range_true': np.max(y_true) - np.min(y_true),
            'score_range_pred': np.max(y_pred) - np.min(y_pred),
            'score_std_true': np.std(y_true),
            'score_std_pred': np.std(y_pred),
        }
        
        # Calculate accuracy within different thresholds
        thresholds = [0.5, 1.0, 1.5, 2.0]
        for threshold in thresholds:
            accuracy = np.mean(np.abs(y_true - y_pred) <= threshold)
            metrics[f'accuracy_within_{threshold}'] = accuracy
        
        # Calculate score category accuracy
        def get_score_category(score):
            if score >= 8.5: return 'Excellent'
            elif score >= 7.0: return 'Good'
            elif score >= 5.5: return 'Average'
            else: return 'Poor'
        
        true_categories = [get_score_category(score) for score in y_true]
        pred_categories = [get_score_category(score) for score in y_pred]
        
        category_accuracy = np.mean([t == p for t, p in zip(true_categories, pred_categories)])
        metrics['category_accuracy'] = category_accuracy
        
        return metrics
    
    def analyze_prediction_errors(self, y_true: List[float], y_pred: List[float], 
                                essays: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prediction errors in detail."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        errors = y_pred - y_true
        
        analysis = {
            'error_statistics': {
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'std_error': np.std(errors),
                'min_error': np.min(errors),
                'max_error': np.max(errors),
                'q25_error': np.percentile(errors, 25),
                'q75_error': np.percentile(errors, 75),
            },
            
            'error_distribution': {
                'positive_errors': np.sum(errors > 0),
                'negative_errors': np.sum(errors < 0),
                'zero_errors': np.sum(errors == 0),
                'large_errors': np.sum(np.abs(errors) > 2.0),
            },
            
            'worst_predictions': [],
            'best_predictions': [],
        }
        
        # Find worst and best predictions
        error_magnitudes = np.abs(errors)
        worst_indices = np.argsort(error_magnitudes)[-5:]  # Top 5 worst
        best_indices = np.argsort(error_magnitudes)[:5]    # Top 5 best
        
        for idx in worst_indices:
            analysis['worst_predictions'].append({
                'index': int(idx),
                'title': essays[idx]['title'] if idx < len(essays) else f'Essay {idx}',
                'true_score': float(y_true[idx]),
                'predicted_score': float(y_pred[idx]),
                'error': float(errors[idx]),
                'error_magnitude': float(error_magnitudes[idx])
            })
        
        for idx in best_indices:
            analysis['best_predictions'].append({
                'index': int(idx),
                'title': essays[idx]['title'] if idx < len(essays) else f'Essay {idx}',
                'true_score': float(y_true[idx]),
                'predicted_score': float(y_pred[idx]),
                'error': float(errors[idx]),
                'error_magnitude': float(error_magnitudes[idx])
            })
        
        return analysis
    
    def analyze_feature_importance(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance and correlations."""
        if features_df is None or features_df.empty:
            return {}
        
        # Select numeric features
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col not in ['essay_id', 'score']]
        
        if len(numeric_features) == 0:
            return {}
        
        # Calculate correlations with score
        correlations = {}
        for feature in numeric_features:
            try:
                corr = features_df[feature].corr(features_df['score'])
                if not np.isnan(corr):
                    correlations[feature] = abs(corr)
            except:
                continue
        
        # Sort by correlation strength
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Feature importance analysis
        importance_analysis = {
            'top_features': dict(sorted_features[:20]),
            'feature_categories': {
                'readability': {},
                'vocabulary': {},
                'grammar': {},
                'structure': {},
                'sentiment': {}
            },
            'correlation_matrix': None
        }
        
        # Categorize features
        for feature, importance in sorted_features:
            if 'readability' in feature.lower() or 'flesch' in feature.lower():
                importance_analysis['feature_categories']['readability'][feature] = importance
            elif 'vocabulary' in feature.lower() or 'lexical' in feature.lower() or 'unique' in feature.lower():
                importance_analysis['feature_categories']['vocabulary'][feature] = importance
            elif 'noun' in feature.lower() or 'verb' in feature.lower() or 'grammar' in feature.lower():
                importance_analysis['feature_categories']['grammar'][feature] = importance
            elif 'sentence' in feature.lower() or 'paragraph' in feature.lower() or 'structure' in feature.lower():
                importance_analysis['feature_categories']['structure'][feature] = importance
            elif 'sentiment' in feature.lower() or 'positive' in feature.lower() or 'negative' in feature.lower():
                importance_analysis['feature_categories']['sentiment'][feature] = importance
        
        # Create correlation matrix for top features
        top_features = list(importance_analysis['top_features'].keys())[:10]
        if len(top_features) > 1:
            corr_matrix = features_df[top_features + ['score']].corr()
            importance_analysis['correlation_matrix'] = corr_matrix.to_dict()
        
        return importance_analysis
    
    def create_evaluation_report(self, trainer: EssayScoringTrainer, 
                               essays: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive evaluation report."""
        logger.info("Creating comprehensive evaluation report...")
        
        # Prepare dataset
        train_dataset, test_dataset, X_test, y_test = trainer.prepare_dataset(essays)
        
        # Get predictions
        predictions = trainer.predict(X_test)
        
        # Calculate performance metrics
        performance_metrics = self.evaluate_model_performance(y_test, predictions)
        
        # Analyze prediction errors
        error_analysis = self.analyze_prediction_errors(y_test, predictions, X_test)
        
        # Analyze features
        features_df = analyze_essay_features(essays)
        feature_analysis = self.analyze_feature_importance(features_df)
        
        # Create comprehensive report
        report = {
            'model_info': {
                'model_name': trainer.model_name,
                'total_essays': len(essays),
                'training_essays': len(essays) - len(y_test),
                'test_essays': len(y_test),
                'max_length': config.max_length,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'num_epochs': config.num_epochs
            },
            'performance_metrics': performance_metrics,
            'error_analysis': error_analysis,
            'feature_analysis': feature_analysis,
            'recommendations': self.generate_recommendations(performance_metrics, error_analysis)
        }
        
        return report
    
    def generate_recommendations(self, metrics: Dict[str, float], 
                               error_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Performance-based recommendations
        if metrics['r2'] < 0.5:
            recommendations.append("Low R² score indicates poor model fit. Consider collecting more training data or adjusting model architecture.")
        
        if metrics['mape'] > 20:
            recommendations.append("High MAPE suggests significant prediction errors. Review feature engineering and model complexity.")
        
        if metrics['category_accuracy'] < 0.7:
            recommendations.append("Low category accuracy indicates difficulty in distinguishing score ranges. Consider ensemble methods or different model architectures.")
        
        # Error-based recommendations
        if error_analysis['error_statistics']['std_error'] > 1.5:
            recommendations.append("High error variance suggests inconsistent predictions. Consider regularization techniques or more robust features.")
        
        if error_analysis['error_distribution']['large_errors'] > len(error_analysis['worst_predictions']) * 0.3:
            recommendations.append("Many large errors detected. Review outlier detection and consider robust loss functions.")
        
        # General recommendations
        recommendations.extend([
            "Consider implementing cross-validation for more robust model evaluation.",
            "Analyze feature importance to identify the most predictive features.",
            "Experiment with different transformer models (BERT, RoBERTa, DeBERTa) for potentially better performance.",
            "Implement data augmentation techniques to increase training data diversity.",
            "Consider ensemble methods combining multiple models for improved accuracy."
        ])
        
        return recommendations
    
    def save_evaluation_report(self, report: Dict[str, Any], 
                             output_path: Optional[Path] = None) -> Path:
        """Save evaluation report to file."""
        if output_path is None:
            output_path = config.logs_dir / f"evaluation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_path}")
        return output_path
    
    def create_visualizations(self, report: Dict[str, Any], 
                            output_dir: Optional[Path] = None) -> List[Path]:
        """Create visualization plots for the evaluation report."""
        if output_dir is None:
            output_dir = config.logs_dir / "evaluation_plots"
        
        output_dir.mkdir(exist_ok=True)
        
        plot_paths = []
        
        # Performance metrics visualization
        metrics = report['performance_metrics']
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Regression Metrics', 'Accuracy Metrics', 'Correlation Metrics', 'Distribution Metrics'],
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Regression metrics
        reg_metrics = ['mse', 'mae', 'rmse', 'r2']
        reg_values = [metrics[m] for m in reg_metrics]
        fig.add_trace(go.Bar(x=reg_metrics, y=reg_values, name='Regression'), row=1, col=1)
        
        # Accuracy metrics
        acc_metrics = [f'accuracy_within_{t}' for t in [0.5, 1.0, 1.5, 2.0]]
        acc_values = [metrics[m] for m in acc_metrics]
        fig.add_trace(go.Bar(x=[0.5, 1.0, 1.5, 2.0], y=acc_values, name='Accuracy'), row=1, col=2)
        
        # Correlation metrics
        corr_metrics = ['pearson_correlation', 'spearman_correlation']
        corr_values = [metrics[m] for m in corr_metrics]
        fig.add_trace(go.Bar(x=['Pearson', 'Spearman'], y=corr_values, name='Correlation'), row=2, col=1)
        
        # Distribution metrics
        dist_metrics = ['score_std_true', 'score_std_pred']
        dist_values = [metrics[m] for m in dist_metrics]
        fig.add_trace(go.Bar(x=['True Scores', 'Predicted Scores'], y=dist_values, name='Std Dev'), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Model Performance Overview")
        plot_path = output_dir / "performance_overview.html"
        fig.write_html(str(plot_path))
        plot_paths.append(plot_path)
        
        # Feature importance plot
        if 'feature_analysis' in report and 'top_features' in report['feature_analysis']:
            top_features = report['feature_analysis']['top_features']
            if top_features:
                fig = px.bar(
                    x=list(top_features.values()),
                    y=list(top_features.keys()),
                    orientation='h',
                    title="Top 20 Most Important Features",
                    labels={'x': 'Correlation with Score', 'y': 'Features'}
                )
                fig.update_layout(height=600)
                plot_path = output_dir / "feature_importance.html"
                fig.write_html(str(plot_path))
                plot_paths.append(plot_path)
        
        logger.info(f"Created {len(plot_paths)} visualization plots")
        return plot_paths

def run_comprehensive_evaluation():
    """Run comprehensive model evaluation."""
    logger.info("Starting comprehensive model evaluation...")
    
    # Initialize components
    db = EssayDatabase()
    evaluator = ModelEvaluator()
    
    # Get essays
    essays = db.get_all_essays()
    if len(essays) < 5:
        logger.error("Not enough essays for evaluation. Need at least 5 essays.")
        return None
    
    # Initialize and train model
    trainer = EssayScoringTrainer()
    train_dataset, test_dataset, X_test, y_test = trainer.prepare_dataset(essays)
    trainer.train(train_dataset, test_dataset)
    
    # Create evaluation report
    report = evaluator.create_evaluation_report(trainer, essays)
    
    # Save report
    report_path = evaluator.save_evaluation_report(report)
    
    # Create visualizations
    plot_paths = evaluator.create_visualizations(report)
    
    # Print summary
    print("\n" + "="*60)
    print("🧠 COMPREHENSIVE MODEL EVALUATION REPORT")
    print("="*60)
    
    print(f"\n📊 Model Performance Summary:")
    print(f"   R² Score: {report['performance_metrics']['r2']:.3f}")
    print(f"   Mean Absolute Error: {report['performance_metrics']['mae']:.3f}")
    print(f"   Root Mean Squared Error: {report['performance_metrics']['rmse']:.3f}")
    print(f"   Mean Absolute Percentage Error: {report['performance_metrics']['mape']:.1f}%")
    print(f"   Category Accuracy: {report['performance_metrics']['category_accuracy']:.1%}")
    
    print(f"\n📈 Error Analysis:")
    print(f"   Mean Error: {report['error_analysis']['error_statistics']['mean_error']:.3f}")
    print(f"   Error Standard Deviation: {report['error_analysis']['error_statistics']['std_error']:.3f}")
    print(f"   Large Errors (>2.0): {report['error_analysis']['error_distribution']['large_errors']}")
    
    print(f"\n🔍 Top 5 Most Important Features:")
    if 'feature_analysis' in report and 'top_features' in report['feature_analysis']:
        for i, (feature, importance) in enumerate(list(report['feature_analysis']['top_features'].items())[:5]):
            print(f"   {i+1}. {feature}: {importance:.3f}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(report['recommendations'][:5]):
        print(f"   {i+1}. {rec}")
    
    print(f"\n📁 Files Generated:")
    print(f"   Report: {report_path}")
    for plot_path in plot_paths:
        print(f"   Plot: {plot_path}")
    
    return report

if __name__ == "__main__":
    # Run comprehensive evaluation
    report = run_comprehensive_evaluation()
    
    if report:
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 Model shows {'good' if report['performance_metrics']['r2'] > 0.7 else 'moderate' if report['performance_metrics']['r2'] > 0.5 else 'poor'} performance")
    else:
        print(f"\n❌ Evaluation failed. Please check the logs for details.")
