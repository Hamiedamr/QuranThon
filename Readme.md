# QuranThon Example

## Installation

- ```docker pull qdrant/qdrant```
- ```docker run -p 6333:6333 -v qdrant_storage:<local path> qdrant/qdrant```
- ```pip install -r requirments.txt```

## Run

- ```uvicorn app:app --reload```
- open [localhost:8000/docs](localhost:8000/docs)
