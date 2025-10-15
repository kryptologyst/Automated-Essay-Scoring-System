import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import random
from config import config, logger

Base = declarative_base()

class Essay(Base):
    """Database model for essays."""
    __tablename__ = 'essays'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    score = Column(Float, nullable=False)
    prompt = Column(Text, nullable=True)
    grade_level = Column(String(20), nullable=True)
    word_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'score': self.score,
            'prompt': self.prompt,
            'grade_level': self.grade_level,
            'word_count': self.word_count,
            'created_at': self.created_at.isoformat()
        }

class EssayDatabase:
    """Database manager for essay scoring system."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or config.database_url
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"Database initialized: {self.database_url}")
    
    def get_session(self):
        """Get database session."""
        return self.SessionLocal()
    
    def add_essay(self, title: str, content: str, score: float, 
                  prompt: str = None, grade_level: str = None) -> int:
        """Add a new essay to the database."""
        with self.get_session() as session:
            essay = Essay(
                title=title,
                content=content,
                score=score,
                prompt=prompt,
                grade_level=grade_level,
                word_count=len(content.split())
            )
            session.add(essay)
            session.commit()
            session.refresh(essay)
            logger.info(f"Added essay: {title} (Score: {score})")
            return essay.id
    
    def get_essay(self, essay_id: int) -> Optional[Dict[str, Any]]:
        """Get essay by ID."""
        with self.get_session() as session:
            essay = session.query(Essay).filter(Essay.id == essay_id).first()
            return essay.to_dict() if essay else None
    
    def get_all_essays(self) -> List[Dict[str, Any]]:
        """Get all essays."""
        with self.get_session() as session:
            essays = session.query(Essay).all()
            return [essay.to_dict() for essay in essays]
    
    def get_essays_by_score_range(self, min_score: float, max_score: float) -> List[Dict[str, Any]]:
        """Get essays within score range."""
        with self.get_session() as session:
            essays = session.query(Essay).filter(
                Essay.score >= min_score,
                Essay.score <= max_score
            ).all()
            return [essay.to_dict() for essay in essays]
    
    def update_essay_score(self, essay_id: int, new_score: float) -> bool:
        """Update essay score."""
        with self.get_session() as session:
            essay = session.query(Essay).filter(Essay.id == essay_id).first()
            if essay:
                essay.score = new_score
                session.commit()
                logger.info(f"Updated essay {essay_id} score to {new_score}")
                return True
            return False
    
    def delete_essay(self, essay_id: int) -> bool:
        """Delete essay by ID."""
        with self.get_session() as session:
            essay = session.query(Essay).filter(Essay.id == essay_id).first()
            if essay:
                session.delete(essay)
                session.commit()
                logger.info(f"Deleted essay {essay_id}")
                return True
            return False

def create_mock_data():
    """Create mock essay data for testing."""
    mock_essays = [
        {
            "title": "The Importance of Education",
            "content": "Education is the foundation of a prosperous society. It empowers individuals with knowledge, critical thinking skills, and the ability to contribute meaningfully to their communities. In today's rapidly changing world, education has become more important than ever before. It not only provides job opportunities but also helps people understand complex issues, make informed decisions, and participate actively in democratic processes. Quality education should be accessible to everyone, regardless of their socioeconomic background, as it is a fundamental human right that can break the cycle of poverty and create a more equitable world.",
            "score": 8.5,
            "prompt": "Discuss the importance of education in modern society",
            "grade_level": "high_school"
        },
        {
            "title": "Climate Change and Our Future",
            "content": "Climate change represents one of the most pressing challenges of our time. The scientific evidence is overwhelming: human activities, particularly the burning of fossil fuels, have significantly increased greenhouse gas concentrations in the atmosphere. This has led to rising global temperatures, melting ice caps, sea level rise, and more frequent extreme weather events. The consequences are already visible worldwide, affecting ecosystems, agriculture, and human settlements. Addressing climate change requires immediate and coordinated action at all levels - individual, national, and international. We must transition to renewable energy sources, implement sustainable practices, and develop innovative technologies to mitigate and adapt to these changes.",
            "score": 9.2,
            "prompt": "Explain the causes and effects of climate change",
            "grade_level": "high_school"
        },
        {
            "title": "The Role of Technology in Communication",
            "content": "Technology has revolutionized how we communicate with each other. Social media platforms, instant messaging apps, and video conferencing tools have made it possible to stay connected with people across the globe. While these advances have brought many benefits, they have also created new challenges. The digital divide means that not everyone has equal access to these technologies. Additionally, the rise of social media has led to concerns about privacy, misinformation, and the impact on mental health. It's important to use technology responsibly and maintain a balance between digital and face-to-face interactions.",
            "score": 7.8,
            "prompt": "How has technology changed communication?",
            "grade_level": "middle_school"
        },
        {
            "title": "Benefits of Physical Exercise",
            "content": "Regular physical exercise is essential for maintaining good health. It strengthens the heart, improves circulation, and helps prevent chronic diseases like diabetes and hypertension. Exercise also has mental health benefits, reducing stress and anxiety while improving mood and cognitive function. Whether it's walking, swimming, cycling, or playing sports, any form of physical activity can contribute to overall well-being. It's recommended that adults get at least 150 minutes of moderate-intensity exercise per week. Making exercise a regular part of your routine can lead to a longer, healthier life.",
            "score": 8.0,
            "prompt": "Describe the benefits of regular exercise",
            "grade_level": "middle_school"
        },
        {
            "title": "The Value of Friendship",
            "content": "Friendship is one of life's greatest treasures. True friends provide emotional support, share in our joys and sorrows, and help us grow as individuals. They offer different perspectives, challenge our thinking, and encourage us to be our best selves. Building and maintaining friendships requires effort, trust, and mutual respect. In our busy lives, it's important to make time for friends and nurture these relationships. Good friends can last a lifetime and provide a sense of belonging and community that enriches our lives in countless ways.",
            "score": 7.5,
            "prompt": "What makes friendship valuable?",
            "grade_level": "middle_school"
        },
        {
            "title": "Space Exploration and Human Progress",
            "content": "Space exploration represents humanity's quest to understand the universe and push the boundaries of what's possible. From the first satellite launch to the International Space Station, space missions have led to numerous technological advances that benefit life on Earth. GPS navigation, weather forecasting, and satellite communications are just a few examples of space technology applications. Future missions to Mars and beyond could unlock new resources and potentially new places for human habitation. While space exploration is expensive and risky, the knowledge gained and technologies developed make it a worthwhile investment in our future.",
            "score": 8.8,
            "prompt": "Discuss the importance of space exploration",
            "grade_level": "high_school"
        },
        {
            "title": "The Impact of Social Media on Youth",
            "content": "Social media has become an integral part of young people's lives, offering both opportunities and challenges. On one hand, it provides platforms for self-expression, learning, and connecting with others who share similar interests. It can also be a powerful tool for social activism and raising awareness about important issues. However, social media also presents risks such as cyberbullying, privacy concerns, and the pressure to present a perfect image online. The constant comparison with others can negatively impact self-esteem and mental health. It's crucial for young people to use social media mindfully and for parents and educators to provide guidance on digital literacy and online safety.",
            "score": 8.3,
            "prompt": "Analyze the impact of social media on young people",
            "grade_level": "high_school"
        },
        {
            "title": "Why Reading Matters",
            "content": "Reading is a fundamental skill that opens doors to knowledge, imagination, and personal growth. Through books, we can explore different cultures, historical periods, and perspectives without leaving our homes. Reading improves vocabulary, writing skills, and critical thinking abilities. It also enhances empathy by allowing us to experience life through different characters' eyes. In our digital age, it's important to maintain the habit of reading books, as it provides deeper engagement than scrolling through social media. Libraries and bookstores remain valuable community resources that promote literacy and lifelong learning.",
            "score": 7.9,
            "prompt": "Explain why reading is important",
            "grade_level": "middle_school"
        },
        {
            "title": "The Future of Artificial Intelligence",
            "content": "Artificial Intelligence is rapidly transforming various industries and aspects of daily life. From virtual assistants to autonomous vehicles, AI technologies are becoming increasingly sophisticated and widespread. While AI offers tremendous potential for improving efficiency, healthcare, and scientific research, it also raises important questions about job displacement, privacy, and ethical considerations. The development of AI must be guided by principles that ensure it benefits humanity while minimizing potential risks. As we move forward, it's crucial to invest in education and training programs that prepare people for an AI-driven economy and to establish regulations that promote responsible AI development.",
            "score": 9.0,
            "prompt": "Discuss the future implications of artificial intelligence",
            "grade_level": "high_school"
        },
        {
            "title": "The Importance of Cultural Diversity",
            "content": "Cultural diversity enriches our communities and societies in countless ways. Different cultures bring unique perspectives, traditions, and knowledge that contribute to innovation and creativity. Exposure to diverse cultures helps us develop empathy, tolerance, and a broader understanding of the world. In our increasingly globalized society, cultural diversity is not just valuable but essential for peaceful coexistence. Schools and workplaces should celebrate and promote cultural diversity through inclusive practices and multicultural education. By embracing our differences, we can build stronger, more resilient communities that benefit everyone.",
            "score": 8.6,
            "prompt": "Why is cultural diversity important?",
            "grade_level": "high_school"
        }
    ]
    
    return mock_essays

def initialize_database():
    """Initialize database with mock data."""
    db = EssayDatabase()
    
    # Check if database already has data
    existing_essays = db.get_all_essays()
    if existing_essays:
        logger.info(f"Database already contains {len(existing_essays)} essays")
        return db
    
    # Add mock data
    mock_essays = create_mock_data()
    for essay_data in mock_essays:
        db.add_essay(**essay_data)
    
    logger.info(f"Initialized database with {len(mock_essays)} mock essays")
    return db

if __name__ == "__main__":
    # Initialize database with mock data
    db = initialize_database()
    
    # Display sample data
    essays = db.get_all_essays()
    print(f"\n📚 Database contains {len(essays)} essays:")
    for essay in essays[:3]:  # Show first 3 essays
        print(f"\n📝 {essay['title']}")
        print(f"   Score: {essay['score']}")
        print(f"   Grade Level: {essay['grade_level']}")
        print(f"   Word Count: {essay['word_count']}")
        print(f"   Content Preview: {essay['content'][:100]}...")
