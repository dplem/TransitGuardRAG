"""
Batch CSV to Pinecone Loader
Processes all CSV files in a data folder and uploads them to Pinecone with embeddings
"""

import pandas as pd
import os
import glob
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import uuid
from tqdm import tqdm
import time
from dotenv import load_dotenv
import hashlib
from pathlib import Path

class BatchCSVToPinecone:
    def __init__(self, data_folder="data", index_name="csv-embeddings"):
        """
        Initialize the batch CSV loader
        
        Args:
            data_folder: Folder containing CSV files
            index_name: Name of the Pinecone index to create/use
        """
        # Load environment variables
        load_dotenv()
        
        self.data_folder = data_folder
        self.index_name = index_name
        
        # Get Pinecone API key from .env
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in .env file")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Initialize embedding model (free)
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384
        
        print(f"Initialized with data folder: {data_folder}")
        print(f"Target index: {index_name}")
    
    def get_csv_files(self):
        """Get all CSV files in the data folder"""
        csv_pattern = os.path.join(self.data_folder, "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_folder} folder")
        
        print(f"Found {len(csv_files)} CSV files:")
        for file in csv_files:
            print(f"  - {os.path.basename(file)}")
        
        return csv_files
    
    def create_or_get_index(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print("Waiting for index to be ready...")
            time.sleep(15)  # Wait for index to be ready
        else:
            print(f"Using existing index: {self.index_name}")
        
        return self.pc.Index(self.index_name)
    
    def process_csv_file(self, csv_path):
        """
        Process a single CSV file and prepare documents
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of document dictionaries
        """
        filename = os.path.basename(csv_path)
        print(f"\nProcessing: {filename}")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
            
            # Handle missing values
            df = df.fillna('')
            
            documents = []
            
            for idx, row in df.iterrows():
                # Combine all non-empty columns into text content
                text_parts = []
                metadata = {
                    'source_file': filename,
                    'source_path': csv_path,
                    'row_index': idx
                }
                
                for col, value in row.items():
                    # Convert value to string and clean it
                    str_value = str(value).strip()
                    
                    if str_value and str_value.lower() not in ['nan', 'none', '']:
                        text_parts.append(f"{col}: {str_value}")
                    
                    # Store all columns as metadata (truncate long values)
                    metadata[f"col_{col}"] = str_value[:200] if len(str_value) > 200 else str_value
                
                # Create text content for embedding
                text_content = " | ".join(text_parts)
                
                if text_content.strip():  # Only process non-empty content
                    # Create unique ID based on file and row
                    unique_string = f"{filename}_{idx}_{text_content[:100]}"
                    doc_id = hashlib.md5(unique_string.encode()).hexdigest()
                    
                    documents.append({
                        'id': doc_id,
                        'text': text_content,
                        'metadata': metadata
                    })
            
            print(f"  Prepared {len(documents)} documents")
            return documents
            
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
            return []
    
    def generate_embeddings_batch(self, texts, batch_size=32):
        """Generate embeddings in batches"""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            embeddings.extend(batch_embeddings.tolist())
        
        return embeddings
    
    def upload_documents(self, index, documents, batch_size=50):
        """Upload documents with embeddings to Pinecone"""
        if not documents:
            print("No documents to upload")
            return
        
        print(f"Generating embeddings for {len(documents)} documents...")
        texts = [doc['text'] for doc in documents]
        embeddings = self.generate_embeddings_batch(texts)
        
        print("Uploading to Pinecone...")
        vectors = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vectors.append({
                'id': doc['id'],
                'values': embedding,
                'metadata': doc['metadata']
            })
            
            # Upload in batches
            if len(vectors) >= batch_size or i == len(documents) - 1:
                try:
                    index.upsert(vectors=vectors)
                    vectors = []
                    time.sleep(0.1)  # Rate limiting
                except Exception as e:
                    print(f"Error uploading batch: {e}")
        
        print(f"Successfully uploaded {len(documents)} documents")
    
    def process_all_csv_files(self, batch_size=50):
        """Process all CSV files in the data folder"""
        # Get all CSV files
        csv_files = self.get_csv_files()
        
        # Create or get index
        index = self.create_or_get_index()
        
        # Process each CSV file
        total_documents = 0
        
        for csv_file in csv_files:
            documents = self.process_csv_file(csv_file)
            if documents:
                self.upload_documents(index, documents, batch_size)
                total_documents += len(documents)
        
        print(f"\n=== SUMMARY ===")
        print(f"Processed {len(csv_files)} CSV files")
        print(f"Total documents uploaded: {total_documents}")
        print(f"Index name: {self.index_name}")
        
        return index, total_documents
    
    def test_search(self, query="test search", top_k=3):
        """Test search functionality"""
        index = self.pc.Index(self.index_name)
        
        print(f"\nTesting search with query: '{query}'")
        
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()
        
        # Search
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        print(f"Found {len(results.matches)} results:")
        for i, match in enumerate(results.matches):
            print(f"\nResult {i+1}:")
            print(f"  Score: {match.score:.4f}")
            print(f"  Source: {match.metadata.get('source_file', 'Unknown')}")
            print(f"  Row: {match.metadata.get('row_index', 'Unknown')}")
            # Show first few metadata fields
            text_preview = ""
            for key, value in match.metadata.items():
                if key.startswith('col_') and value:
                    text_preview += f"{value} "
                    if len(text_preview) > 150:
                        break
            print(f"  Preview: {text_preview[:150]}...")

def main():
    """Main function to run the batch CSV processor"""
    
    # Configuration
    DATA_FOLDER = "data"  # Folder containing CSV files
    INDEX_NAME = "csv-embeddings"  # Pinecone index name
    
    print("=== Batch CSV to Pinecone Loader ===")
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Index name: {INDEX_NAME}")
    
    # Check if data folder exists
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Data folder '{DATA_FOLDER}' not found!")
        print("Please create the folder and add your CSV files.")
        return
    
    try:
        # Initialize processor
        processor = BatchCSVToPinecone(
            data_folder=DATA_FOLDER,
            index_name=INDEX_NAME
        )
        
        # Process all CSV files
        index, total_docs = processor.process_all_csv_files()
        
        if total_docs > 0:
            # Test search
            test_query = input("\nEnter a test search query (or press Enter to skip): ").strip()
            if test_query:
                processor.test_search(test_query)
            
            print(f"\n✅ Successfully processed all CSV files!")
            print(f"Your data is now searchable in Pinecone index: {INDEX_NAME}")
        else:
            print("❌ No documents were processed. Check your CSV files.")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()