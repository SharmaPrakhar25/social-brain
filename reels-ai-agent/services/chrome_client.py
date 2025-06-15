import logging
from typing import Optional
import chromadb
from chromadb.api.models.Collection import Collection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_COLLECTION_NAME = "reel_summaries"
CHROMA_DB_PATH = "./chroma_db"

def get_collection(name: str = DEFAULT_COLLECTION_NAME) -> Collection:
    """
    Retrieve or create a persistent Chroma collection.
    
    Args:
        name (str, optional): Name of the collection. 
        Defaults to "reel_summaries".
    
    Returns:
        Collection: A persistent Chroma collection.
    
    Raises:
        RuntimeError: If collection creation fails.
    """
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection(name)
        logger.info(f"Successfully retrieved/created collection: {name}")
        return collection
    except Exception as e:
        logger.error(f"Failed to create ChromaDB collection: {e}")
        raise RuntimeError(f"Could not create ChromaDB collection: {e}")