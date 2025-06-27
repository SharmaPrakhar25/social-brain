from fastapi import APIRouter
from fastapi.responses import JSONResponse
from chromadb import PersistentClient

router = APIRouter()

@router.get("/inspect")
def inspect_collection():
    try:
        chroma_client = PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection("reel_summaries")
        results = collection.get(include=["documents", "metadatas"])

        data = [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        return JSONResponse(content={"entries": data})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)