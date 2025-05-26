import requests
from sentence_transformers import SentenceTransformer

# The question to embed and query
#question = "What are the stations near me?"
#question = "What are the total number of crimes today?"
#question = "What are the total number of traffic accidents today?"
question = "What is the safest line in the last 7 days?"

# Load a sentence-transformers model (use the same model as used for Pinecone embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(question).tolist()

response = requests.post(
    "http://localhost:8000/query",
    json={"embedding": embedding, "top_k": 5, "question": question}
)

try:
    data = response.json()
    print(data.get("answer"))
    #if "sources" in data:
    #    print("Sources:")
    #    for src in data["sources"]:
    #        print(src)
except Exception as e:
    print(f"Error decoding JSON: {e}")
    print(f"Status code: {response.status_code}")
    print(f"Response text: {response.text}") 