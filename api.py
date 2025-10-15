from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import logging
from pathlib import Path
import json
import asyncio
from datetime import datetime

from config import config, logger
from database import EssayDatabase, Essay
from models import EssayScoringTrainer
from preprocessing import FeatureExtractor, analyze_essay_features

# Initialize FastAPI app
app = FastAPI(
    title="Automated Essay Scoring System",
    description="A modern AI-powered essay scoring system using transformer models",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables
db = EssayDatabase()
trainer = None
feature_extractor = FeatureExtractor()

# Pydantic models for API
class EssayInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=10, max_length=10000)
    prompt: Optional[str] = Field(None, max_length=1000)
    grade_level: Optional[str] = Field(None, regex="^(elementary|middle_school|high_school|college)$")

class EssayScoreRequest(BaseModel):
    content: str = Field(..., min_length=10, max_length=10000)
    title: Optional[str] = Field(None, max_length=200)

class EssayResponse(BaseModel):
    id: int
    title: str
    content: str
    score: float
    prompt: Optional[str]
    grade_level: Optional[str]
    word_count: int
    created_at: str

class ScoreResponse(BaseModel):
    score: float
    confidence: float
    features: Dict[str, Any]
    analysis: Dict[str, Any]

class TrainingResponse(BaseModel):
    status: str
    message: str
    metrics: Optional[Dict[str, float]]

# Dependency to get database session
def get_database():
    return db

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/essays", response_model=List[EssayResponse])
async def get_essays(skip: int = 0, limit: int = 100):
    """Get all essays with pagination."""
    essays = db.get_all_essays()
    return essays[skip:skip + limit]

@app.get("/api/essays/{essay_id}", response_model=EssayResponse)
async def get_essay(essay_id: int):
    """Get a specific essay by ID."""
    essay = db.get_essay(essay_id)
    if not essay:
        raise HTTPException(status_code=404, detail="Essay not found")
    return essay

@app.post("/api/essays", response_model=EssayResponse)
async def create_essay(essay: EssayInput, database: EssayDatabase = Depends(get_database)):
    """Create a new essay."""
    try:
        essay_id = database.add_essay(
            title=essay.title,
            content=essay.content,
            score=0.0,  # Will be scored separately
            prompt=essay.prompt,
            grade_level=essay.grade_level
        )
        
        # Get the created essay
        created_essay = database.get_essay(essay_id)
        return created_essay
    except Exception as e:
        logger.error(f"Error creating essay: {e}")
        raise HTTPException(status_code=500, detail="Failed to create essay")

@app.post("/api/score", response_model=ScoreResponse)
async def score_essay(request: EssayScoreRequest):
    """Score an essay using the AI model."""
    global trainer
    
    try:
        # Initialize trainer if not already done
        if not trainer:
            trainer = EssayScoringTrainer()
            try:
                trainer.load_model()
            except FileNotFoundError:
                raise HTTPException(
                    status_code=503, 
                    detail="Model not trained yet. Please train the model first."
                )
        
        # Predict score
        scores = trainer.predict([request.content])
        predicted_score = scores[0]
        
        # Extract features
        features = feature_extractor.preprocess_and_extract_features(request.content)
        
        # Calculate confidence (simplified)
        confidence = min(0.95, max(0.1, 1.0 - abs(predicted_score - 5.0) / 10.0))
        
        # Generate analysis
        analysis = {
            "word_count": features.get('original_word_count', 0),
            "readability": features.get('original_flesch_reading_ease', 0),
            "vocabulary_diversity": features.get('original_lexical_diversity', 0),
            "avg_word_length": features.get('original_avg_word_length', 0),
            "sentence_count": features.get('original_sentence_count', 0),
            "grammar_score": (
                features.get('original_noun_ratio', 0) + 
                features.get('original_verb_ratio', 0)
            ) * 100
        }
        
        return ScoreResponse(
            score=round(predicted_score, config.score_precision),
            confidence=round(confidence, 2),
            features=features,
            analysis=analysis
        )
        
    except Exception as e:
        logger.error(f"Error scoring essay: {e}")
        raise HTTPException(status_code=500, detail="Failed to score essay")

@app.post("/api/train", response_model=TrainingResponse)
async def train_model():
    """Train the essay scoring model."""
    global trainer
    
    try:
        # Get essays from database
        essays = db.get_all_essays()
        
        if len(essays) < 5:
            raise HTTPException(
                status_code=400, 
                detail="Not enough essays for training. Need at least 5 essays."
            )
        
        # Initialize trainer
        trainer = EssayScoringTrainer()
        
        # Prepare dataset
        train_dataset, test_dataset, X_test, y_test = trainer.prepare_dataset(essays)
        
        # Train model
        trainer.train(train_dataset, test_dataset)
        
        # Evaluate model
        metrics, results = trainer.evaluate(test_dataset, X_test, y_test)
        
        return TrainingResponse(
            status="success",
            message=f"Model trained successfully with {len(essays)} essays",
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")

@app.get("/api/features/{essay_id}")
async def get_essay_features(essay_id: int):
    """Get detailed features for a specific essay."""
    essay = db.get_essay(essay_id)
    if not essay:
        raise HTTPException(status_code=404, detail="Essay not found")
    
    features = feature_extractor.preprocess_and_extract_features(essay['content'])
    return {
        "essay_id": essay_id,
        "title": essay['title'],
        "features": features
    }

@app.get("/api/analytics")
async def get_analytics():
    """Get analytics and statistics."""
    essays = db.get_all_essays()
    
    if not essays:
        return {"message": "No essays found"}
    
    # Calculate statistics
    scores = [essay['score'] for essay in essays]
    word_counts = [essay['word_count'] for essay in essays]
    
    analytics = {
        "total_essays": len(essays),
        "score_statistics": {
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "median": sorted(scores)[len(scores) // 2]
        },
        "word_count_statistics": {
            "mean": sum(word_counts) / len(word_counts),
            "min": min(word_counts),
            "max": max(word_counts),
            "median": sorted(word_counts)[len(word_counts) // 2]
        },
        "grade_level_distribution": {},
        "recent_essays": essays[-5:]  # Last 5 essays
    }
    
    # Grade level distribution
    grade_levels = [essay['grade_level'] for essay in essays if essay['grade_level']]
    grade_counts = {}
    for grade in grade_levels:
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
    analytics["grade_level_distribution"] = grade_counts
    
    return analytics

@app.delete("/api/essays/{essay_id}")
async def delete_essay(essay_id: int):
    """Delete an essay."""
    success = db.delete_essay(essay_id)
    if not success:
        raise HTTPException(status_code=404, detail="Essay not found")
    return {"message": "Essay deleted successfully"}

@app.put("/api/essays/{essay_id}/score")
async def update_essay_score(essay_id: int, score: float):
    """Update an essay's score."""
    if score < config.min_score or score > config.max_score:
        raise HTTPException(
            status_code=400, 
            detail=f"Score must be between {config.min_score} and {config.max_score}"
        )
    
    success = db.update_essay_score(essay_id, score)
    if not success:
        raise HTTPException(status_code=404, detail="Essay not found")
    
    return {"message": "Score updated successfully"}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    # Initialize database with mock data
    logger.info("Initializing database...")
    db = EssayDatabase()
    
    # Check if we have essays, if not, initialize with mock data
    essays = db.get_all_essays()
    if not essays:
        logger.info("No essays found, initializing with mock data...")
        from database import initialize_database
        initialize_database()
    
    # Start the server
    logger.info(f"Starting server on {config.api_host}:{config.api_port}")
    uvicorn.run(
        "api:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug,
        log_level="info"
    )
