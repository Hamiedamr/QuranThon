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
app = FastAPI(lifespan=lifespan, title="QuranThon")

encoder = SentenceTransformer("all-MiniLM-L6-v2")


qdrant = QdrantClient(host="localhost", port=6333)

@app.post("/train")

async def train(
    file: UploadFile = File(...),
):
    '''
    upload json file contains array of objects mainly have description and name attributes
    '''
    contents = await file.read()
    
   
    uploaded_documents = eval(contents.decode("utf-8"))
    
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
            for idx, doc in enumerate(uploaded_documents)
        ],
    )

    return {"message": "Documents uploaded and collection updated successfully"}

@app.post("/search")
async def search(
    query: str,
):
    '''write your search query'''
    query_vector = encoder.encode(query).tolist()
    hits = qdrant.search(
        collection_name="my_books",
        query_vector=query_vector,
        limit=3,
    )

    search_results = [{"payload": hit.payload, "score": hit.score} for hit in hits]
    return {"search_results": search_results}
