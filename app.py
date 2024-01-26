from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    qdrant.recreate_collection(
        collection_name="my_books",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.DOT,
        ),
    )
    yield
app = FastAPI(lifespan=lifespan)

encoder = SentenceTransformer("all-MiniLM-L6-v2")


qdrant = QdrantClient(host="localhost", port=6333)

# Endpoint for uploading documents with authentication
@app.post("/train")
async def train(
    file: UploadFile = File(...),
):
    # Your training logic here (if any)
    contents = await file.read()
    
    # Convert the contents to a list of documents
    uploaded_documents = eval(contents.decode("utf-8"))  # Note: This is just a simple example, ensure security in a production setting
    
    # Update the global documents variable
    global documents
    documents = uploaded_documents

    # Recreate and upload the Qdrant collection with the updated documents
    qdrant.recreate_collection(
        collection_name="my_books",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.DOT,
        ),
    )

    qdrant.upload_records(
        collection_name="my_books",
        records=[
            models.Record(
                id=idx, vector=encoder.encode(doc["description"] +' '+ doc['name']).tolist(), payload=doc
            )
            for idx, doc in enumerate(documents)
        ],
    )

    return {"message": "Documents uploaded and collection updated successfully"}

# Endpoint for searching text with authentication
@app.post("/search")
async def search(
    query: str,
):
    query_vector = encoder.encode(query).tolist()
    hits = qdrant.search(
        collection_name="my_books",
        query_vector=query_vector,
        limit=3,
    )

    search_results = [{"payload": hit.payload, "score": hit.score} for hit in hits]
    return {"search_results": search_results}
