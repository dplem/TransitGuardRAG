# TransitGuardRAG: Pinecone + Claude RAG API

## Overview

This project provides a FastAPI-based API for querying a Pinecone vector database using embedding vectors and generating answers with Anthropic's Claude Haiku 3 model. It includes scripts for embedding questions, querying the API, and is ready for deployment on Railway.

---

## Features
- **/query endpoint:** Accepts a POST request with an embedding vector and question, returns a generated answer and sources.
- **/health endpoint:** Simple health check for the API.
- **Special queries:**
  - Total number of crimes today
  - Total number of traffic accidents today
  - Safest line in the last 7 days
  - Closest/nearby stations
- **Python client script:** Converts a question to an embedding and queries the API (local or Railway).
- **Dockerized & Railway-ready:** Easily build and run the API in a container or deploy to Railway.

---

## Project Structure
```
TransitGuardRAG/
│
├── main.py                  # FastAPI app for Pinecone + Claude querying
├── query_pinecone.py        # Python script to embed a question and query local API
├── railway_query_pinecone.py# Python script to query deployed Railway API
├── requirements.txt
├── Dockerfile
├── start.sh                 # Entrypoint for Railway
├── README.md
└── .env                     # Environment variables (not committed)
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
Create a `.env` file in the root directory with the following:
```
PINECONE_API_KEY=your-pinecone-api-key
CLAUDE_API_KEY=your-anthropic-claude-api-key
```

---

## Running the API Locally
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Docker Deployment

### Build and Run
```bash
docker build -t transitguardrag .
docker run --env-file .env -p 8000:8000 transitguardrag
```

---

## Railway Deployment

1. **Ensure your repo is connected to Railway.**
2. **Set environment variables** (`PINECONE_API_KEY`, `CLAUDE_API_KEY`) in the Railway dashboard.
3. **Dockerfile** and **start.sh** are already configured for Railway:
   - `start.sh` ensures the app listens on the correct port.
   - Dockerfile uses `CMD ["sh", "start.sh"]`.
4. **Deploy!**

---

## API Documentation

### POST `/query`
Query the Pinecone index and generate an answer with Claude.

**Request Body:**
```json
{
  "embedding": [0.1, 0.2, 0.3, ...],
  "top_k": 5,
  "question": "What are the stations near me?"
}
```

**Response:**
```json
{
  "answer": "The stations near your current location are: ...",
  "sources": [ ... ]
}
```

#### Special Queries
- "total number of crimes today"
- "total number of traffic accidents today"
- "safest line in the last 7 days"
- "stations near me" or "closest station"

### GET `/health`
Health check endpoint. Returns `{ "status": "healthy" }`.

---

## Python Client Scripts

### Local API Example: `query_pinecone.py`
Queries a locally running API.

### Railway API Example: `railway_query_pinecone.py`
Queries the deployed Railway API.

```python
import requests
from sentence_transformers import SentenceTransformer

question = "What are the stations near me?"
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(question).tolist()

response = requests.post(
    "https://web-production-1e02.up.railway.app/query",
    json={"embedding": embedding, "top_k": 5, "question": question}
)

print(response.json())
```

---

## Troubleshooting Railway Deployment
- Ensure `start.sh` is executable (`git update-index --chmod=+x start.sh`)
- All required environment variables must be set in Railway dashboard
- Check logs for port or permission errors
- Use `CMD ["sh", "start.sh"]` in Dockerfile

---

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## License
MIT 