import chromadb
from chromadb.utils import embedding_functions
import json
import argparse

# Configuration
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "unsplash_palettes"
MODEL_NAME = "all-mpnet-base-v2"

def main():
    parser = argparse.ArgumentParser(description="Search Unsplash palettes by description.")
    parser.add_argument("query", nargs="?", help="Text query to search for.")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    query = args.query
    if not query:
        query = input("Enter a search query: ")

    results = collection.query(
        query_texts=[query],
        n_results=1
    )

    if results['documents'] and results['documents'][0]:
        doc_json = results['documents'][0][0]
        data = json.loads(doc_json)
        
        print("\n--- Match Found ---")
        print(f"Description: {data['description']}")
        print("Palette:")
        for color in data['palette']:
            print(f"  - {color}")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()
