"""
Content database models
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base
import uuid

# Association table for many-to-many relationship between Content and Keywords
content_keywords = Table(
    'content_keywords',
    Base.metadata,
    Column('content_id', String, ForeignKey('contents.id'), primary_key=True),
    Column('keyword_id', Integer, ForeignKey('keywords.id'), primary_key=True)
)

class Content(Base):
    """
    Main content table for storing processed reels/tweets
    """
    __tablename__ = "contents"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Source information
    source_type = Column(String(20), nullable=False)  # 'instagram', 'twitter', etc.
    original_url = Column(String(500), nullable=False, unique=True)
    title = Column(String(200))
    author = Column(String(100))
    platform_id = Column(String(100))  # Original post ID from platform
    
    # Content data
    transcription = Column(Text)
    summary = Column(Text)
    content_type = Column(String(50))  # 'video', 'audio', 'text'
    duration = Column(Float)  # Duration in seconds for video/audio
    
    # Metadata
    category = Column(String(100))  # Auto-detected category
    sentiment = Column(String(20))  # 'positive', 'negative', 'neutral'
    language = Column(String(10))  # Language code
    
    # Processing metadata
    processing_status = Column(String(20), default='pending')  # 'pending', 'completed', 'failed'
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    published_at = Column(DateTime(timezone=True))  # Original publication date
    
    # Relationships
    keywords = relationship("ContentKeyword", secondary=content_keywords, back_populates="contents")
    
    def __repr__(self):
        return f"<Content(id={self.id}, source_type={self.source_type}, title={self.title})>"

class ContentKeyword(Base):
    """
    Keywords extracted from content
    """
    __tablename__ = "keywords"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    keyword = Column(String(100), nullable=False, unique=True)
    category = Column(String(50))  # 'technology', 'business', 'entertainment', etc.
    frequency = Column(Integer, default=1)  # How often this keyword appears across all content
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    contents = relationship("Content", secondary=content_keywords, back_populates="keywords")
    
    def __repr__(self):
        return f"<ContentKeyword(id={self.id}, keyword={self.keyword}, category={self.category})>" 