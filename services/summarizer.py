"""
Video Summarization Service

This module handles transcription and summarization of video content using:
- Whisper (for transcription)
- BART and LLaMA (for summarization)
"""

import os
import whisper
import yt_dlp
from transformers import pipeline
import uuid
from llama_cpp import Llama
import logging
from typing import Dict
import tempfile
import subprocess
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Print out environment variables and model path
print("Current working directory:", os.getcwd())
print("LLAMA_MODEL_PATH:", os.getenv('LLAMA_MODEL_PATH'))
print("Model file exists:", os.path.exists(os.getenv('LLAMA_MODEL_PATH')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model
try:
    whisper_model = whisper.load_model("base")  # You can change to "small" or "medium" for better quality
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None

# Global LLM model loading with explicit path logging
def load_llama_model():
    model_path = os.getenv('LLAMA_MODEL_PATH')
    logger.info(f"Attempting to load LLaMA model from: {model_path}")
    
    if not model_path:
        logger.error("LLAMA_MODEL_PATH environment variable is not set")
        return None
    
    if not os.path.exists(model_path):
        logger.error(f"LLaMA model file does not exist at: {model_path}")
        return None
    
    try:
        llm = Llama(
            model_path=model_path, 
            n_ctx=2048,
            n_batch=512
        )
        logger.info("LLaMA model loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to load LLaMA model: {e}")
        return None

# Load the LLaMA model
llm = load_llama_model()

def download_video(url: str) -> str:
    """
    Download video from given URL using yt_dlp with comprehensive error handling.
    
    Args:
        url (str): URL of the video to download
    
    Returns:
        str: Path to downloaded audio file
    """
    try:
        # Create a unique temporary directory
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {temp_dir}")

        # Detailed yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s')
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info and download
            info_dict = ydl.extract_info(url, download=True)
            
            # Get the actual downloaded file path
            video_path = ydl.prepare_filename(info_dict)
            
            # Replace extension with .wav
            audio_path = os.path.splitext(video_path)[0] + '.wav'
            
            # Verify file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                raise FileNotFoundError(f"Audio file not created: {audio_path}")
            
            logger.info(f"Successfully downloaded audio to: {audio_path}")
            return audio_path
    
    except Exception as e:
        logger.error(f"Video download failed: {e}")
        
        # Additional debugging: list files in temp directory
        try:
            files = os.listdir(temp_dir)
            logger.info(f"Files in temp directory: {files}")
        except Exception as list_error:
            logger.error(f"Could not list files in temp directory: {list_error}")
        
        raise

def generate_bart_summary(text: str) -> str:
    """
    Generate summary using BART model.
    
    Args:
        text (str): Input text to summarize
    
    Returns:
        str: Summarized text
    """
    # Placeholder for BART summarization
    # You'll need to implement actual BART summarization
    return f"BART summary of: {text[:100]}..."

def generate_llama_summary(text: str) -> str:
    """
    Generate summary using LLaMA model.
    
    Args:
        text (str): Input text to summarize
    
    Returns:
        str: Summarized text
    """
    if not llm:
        return "LLaMA model not available"
    
    prompt = f"Summarize the following text concisely:\n{text}\n\nSummary:"
    
    try:
        response = llm(
            prompt, 
            max_tokens=200, 
            stop=["Human:", "Assistant:"], 
            echo=False
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"LLaMA summarization error: {e}")
        return "Failed to generate LLaMA summary"

def transcribe_and_summarize(url: str) -> Dict[str, str]:
    """
    Transcribe and summarize a video from a given URL.
    
    Args:
        url (str): URL of the video to transcribe and summarize.
    
    Returns:
        Dict[str, str]: Dictionary containing BART and LLaMA summaries.
    """
    audio_path = None
    try:
        # Download video
        audio_path = download_video(url)
        logger.info(f"Downloaded audio: {audio_path}")
        
        # Transcribe video
        if not whisper_model:
            raise ValueError("Whisper model not loaded")
        
        # Add additional file existence check
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        
        # Verify file is not empty
        if os.path.getsize(audio_path) == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        
        result = whisper_model.transcribe(audio_path)
        transcription = result['text']
        logger.info("Transcription completed")
        
        # Generate summaries
        bart_summary = generate_bart_summary(transcription)
        llama_summary = generate_llama_summary(transcription)
        
        return {
            "bart": bart_summary,
            "llama": llama_summary,
            "transcription": transcription
        }
    
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        
        # Log additional context if audio_path exists
        if audio_path and os.path.exists(audio_path):
            logger.info(f"Audio file details:")
            logger.info(f"Path: {audio_path}")
            logger.info(f"Size: {os.path.getsize(audio_path)} bytes")
        
        return {
            "bart": "Summarization failed",
            "llama": "Summarization failed",
            "transcription": "Transcription failed"
        }
    finally:
        # Clean up audio file if it exists
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Deleted temporary audio file: {audio_path}")
            except Exception as cleanup_error:
                logger.error(f"Could not delete audio file: {cleanup_error}")