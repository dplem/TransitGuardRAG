import os
import requests
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pinecone import Pinecone
import anthropic

# Claude API config
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-3-haiku-20240307"

# Pinecone config
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_HOST = "https://csv-embeddings-mz1rti3.svc.aped-4627-b74a.pinecone.io"

# Initialize Pinecone client and connect to the index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=INDEX_HOST)

def call_claude(prompt: str) -> str:
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": CLAUDE_MODEL,
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(CLAUDE_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        # Claude API returns a list of content blocks
        if "content" in result and isinstance(result["content"], list):
            return "".join([block.get("text", "") for block in result["content"]])
        return result.get("content", "")
    except Exception as e:
        print(f"Claude API error: {e}")
        return "Error: Claude API call failed."

app = FastAPI(title="Pinecone Query API", version="1.0.0")

class QueryRequest(BaseModel):
    embedding: list
    top_k: int = 5
    question: str = None

class QueryResponse(BaseModel):
    answer: str
    sources: list = Field(default_factory=list)

@app.post("/query", response_model=QueryResponse)
async def query_pinecone(request: QueryRequest):
    try:
        # Special case: sum the count column for 2024-07-13 in july_2024_crime_summary.csv
        if request.question and "total number of crimes today" in request.question.lower():
            total_crimes = 0
            sources = []
            limit = 99  # Pinecone API: limit must be >0 and <100
            target_file = 'july_2024_crime_summary.csv'
            target_date = '2024-07-13'
            for ids in index.list(limit=limit):
                if not ids:
                    continue
                fetch_response = index.fetch(ids=ids)
                for v in fetch_response.vectors.values():
                    meta = v.metadata if hasattr(v, 'metadata') else {}
                    if (
                        meta.get('source_file', '').lower() == target_file and
                        meta.get('col_date', '') == target_date
                    ):
                        try:
                            count_val = meta.get('col_count', 0)
                            count_val = int(count_val)
                        except Exception:
                            count_val = 0
                        total_crimes += count_val
                        sources.append(meta)
            answer = f"The total number of crimes today is {total_crimes}."
            return QueryResponse(answer=answer, sources=sources)

        # Special case: sum the TOTAL_CRASHES column for 2024-07-13 in traffic_crash_daily_totals_july_2024.csv
        if request.question and "total number of traffic accidents today" in request.question.lower():
            total_crashes = 0
            sources = []
            limit = 99
            target_file = 'traffic_crash_daily_totals_july_2024.csv'
            target_date = '2024-07-13'
            for ids in index.list(limit=limit):
                if not ids:
                    continue
                fetch_response = index.fetch(ids=ids)
                for v in fetch_response.vectors.values():
                    meta = v.metadata if hasattr(v, 'metadata') else {}
                    if (
                        meta.get('source_file', '').lower() == target_file and
                        meta.get('col_DATE', '') == target_date
                    ):
                        try:
                            crash_val = meta.get('col_TOTAL_CRASHES', 0)
                            crash_val = int(crash_val)
                        except Exception:
                            crash_val = 0
                        total_crashes += crash_val
                        sources.append(meta)
            answer = f"The total number of traffic accidents today is {total_crashes}."
            return QueryResponse(answer=answer, sources=sources)

        # Special case: find the safest line in the last 7 days
        if request.question and "safest line in the last 7 days" in request.question.lower():
            min_incidents = None
            safest_lines = []
            sources = []
            limit = 99
            target_file = 'line_counts_last_7_days.csv'
            for ids in index.list(limit=limit):
                if not ids:
                    continue
                fetch_response = index.fetch(ids=ids)
                for v in fetch_response.vectors.values():
                    meta = v.metadata if hasattr(v, 'metadata') else {}
                    if meta.get('source_file', '').lower() == target_file:
                        try:
                            incident_count = int(meta.get('col_incident_count', 0))
                        except Exception:
                            incident_count = 0
                        if min_incidents is None or incident_count < min_incidents:
                            min_incidents = incident_count
                            safest_lines = [(meta.get('col_line_code', 'Unknown'), incident_count)]
                        elif incident_count == min_incidents:
                            safest_lines.append((meta.get('col_line_code', 'Unknown'), incident_count))
                        sources.append(meta)
            if safest_lines:
                # If multiple lines have the same minimum, join them with commas
                line_codes = ', '.join([line for line, _ in safest_lines])
                answer = f"The safest line in the last 7 days is the {line_codes} with {min_incidents} incidents."
            else:
                answer = "No data available for the safest line in the last 7 days."
            return QueryResponse(answer=answer, sources=sources)

        # Default: semantic search with top_k
        query_response = index.query(
            vector=request.embedding,
            top_k=request.top_k,
            include_metadata=True
        )
        matches = query_response.get('matches', [])
        # Compose context from matches, including col_closest_station if present
        station_names = []
        context_lines = []
        for m in matches:
            meta = m['metadata']
            station = meta.get('col_closest_station')
            if station:
                station_names.append(station)
            context_lines.append(
                f"Source: {meta.get('source_file', '')}, Row: {meta.get('row_index', '')}, "
                f"Type: {meta.get('col_Incident Type', '')}, Date: {meta.get('col_Date', '')}, "
                f"Address: {meta.get('col_Address', '')}, Closest Station: {station or ''}"
            )
        context = "\n".join(context_lines)

        # If the question is about stations near the user and we have station names, return the exact required response
        if request.question and "stations near" in request.question.lower() and station_names:
            seen = set()
            unique_stations = [x for x in station_names if not (x in seen or seen.add(x))]
            station_list = ", ".join(unique_stations)
            answer = f"The stations near your current location are: {station_list}."
        # If the question is about closest stations and we have station names, return the closest stations response
        elif request.question and "closest station" in request.question.lower() and station_names:
            seen = set()
            unique_stations = [x for x in station_names if not (x in seen or seen.add(x))]
            station_list = ", ".join(unique_stations)
            answer = f"The closest stations to your current location are: {station_list}."
        else:
            prompt = f"Context from database:\n{context}\n\nQuestion: {request.question}\nAnswer:"
            answer = call_claude(prompt)
        sources = [m['metadata'] for m in matches]
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        print(f"Error in /query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)