# TransitGuardRAG: Pinecone Embedding Query API

## Overview

This project provides a FastAPI-based API for querying a Pinecone vector database using embedding vectors. It also includes a Python script to generate embeddings from natural language questions using `sentence-transformers` and query Pinecone for the most relevant results.

---

## Features
- **/query endpoint:** Accepts a POST request with an embedding vector and returns the top matches from Pinecone.
- **/health endpoint:** Simple health check for the API.
- **Python client script:** Converts a question to an embedding and queries Pinecone for relevant results.
- **Dockerized:** Easily build and run the API in a container.

---

## Project Structure
```
TransitGuardRAG/
│
├── main.py                # FastAPI app for Pinecone querying
├── query_pinecone.py      # Python script to embed a question and query Pinecone
├── requirements.txt
├── Dockerfile
├── README.md
└── .env                   # Environment variables (not committed)
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd TransitGuardRAG
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```
PINECONE_API_KEY=your-pinecone-api-key
```

### 4. Run the API Locally
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Docker Deployment

### Build and Run
```bash
docker build -t rag-csv .
docker run --env-file .env -p 8000:8000 rag-csv
```

---

## API Documentation

### POST `/query`
Query the Pinecone index with an embedding vector.

**Request Body:**
```json
{
  "embedding": [0.1, 0.2, 0.3, ...],
  "top_k": 5
}
```

**Response:**
```json
{
  "matches": [
    {"id": "vec1", "score": 0.95, "metadata": {"source": "..."}},
    ...
  ]
}
```

### GET `/health`
Health check endpoint. Returns `{ "status": "healthy" }`.

---

## Python Client Script: `query_pinecone.py`

This script takes a question, generates an embedding using `sentence-transformers`, and queries the Pinecone API for the most relevant results.

### How to Run the Script

1. **Ensure the API is running** (see above for local or Docker instructions).
2. **Install dependencies** (if not already):
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the script:**
   ```bash
   python query_pinecone.py
   ```

**Example usage in the script:**
```python
import requests
from sentence_transformers import SentenceTransformer

# The question to embed and query
question = "What is the ridership on route 5?"

# Load a sentence-transformers model (use the same model as used for Pinecone embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(question).tolist()

response = requests.post(
    "http://localhost:8000/query",
    json={"embedding": embedding, "top_k": 5}
)
print(response.json())
```

- The embedding model in `query_pinecone.py` should match the model used to generate the embeddings stored in Pinecone.
- The `/query` endpoint only accepts POST requests with a JSON body containing the embedding vector.
- The `/health` endpoint is for health checks and returns a simple status message.
- The API does not provide a root `/` endpoint; accessing it will return a 404.

---

## License
MIT 