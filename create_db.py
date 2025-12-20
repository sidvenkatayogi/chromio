import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from skimage import color
import json
import pickle


# Configuration
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "unsplash_palettes"
MODEL_NAME = "all-mpnet-base-v2"

KD_TREE_PATH = "color_kdtree.pkl"

batch_size = 128

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def main():
    print("Loading prebuilt KDTree...")
    with open(KD_TREE_PATH, "rb") as f:
        data = pickle.load(f)

    tree = data["tree"]
    color_ref_metadata = data["metadata"]

    print("KDTree loaded successfully.")

    # Load Unsplash dataset
    print("Loading Unsplash dataset...")
    unsplash_dataset = load_dataset("1aurent/unsplash-lite-palette", split="train")
    print(f"Loaded {len(unsplash_dataset)} entries.")

    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Create embedding function
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    
    # Get or create collection
    # Force delete if exists to start fresh? No, assume append or fresh.
    # client.delete_collection(COLLECTION_NAME) 
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

    print("Processing documents and populating database...")
    
    documents = []
    metadatas = []
    ids = []
    
    # Iterate with index to use as ID
    for idx, entry in enumerate(tqdm(unsplash_dataset)):

        # Remove try/except to see errors, or print them
        try:
            ai_description = entry.get('ai_description')
            if not ai_description:
                # if idx < 5: print(f"Skipping {idx}: no description")
                continue
                
            palettes = entry.get('palettes')
            if not palettes:
                # if idx < 5: print(f"Skipping {idx}: no palettes")
                continue

            # Extract colors
            formatted_palette = []
            
            # ATTEMPT 1: User instructions "indices 1 through 5".
            # Tracing: Key '1' -> [[r,g,b]]. Key '2' -> [[r,g,b], ...]. 
            # We take the first color from each key as per previous logic.
            
            # Let's inspect the first one
            if idx == 0:
                print(f"Sample palettes keys: {palettes.keys()}")
            
            # Use keys 1 to 5 to extract one color each
            for i in range(1, 6):
                key = str(i)
                if key in palettes:
                    val = palettes[key]
                    rgb_val = None
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
                         rgb_val = val[0]
                    # elif isinstance(val, list) and len(val) == 3 and isinstance(val[0], (int, float)):
                    #      rgb_val = val
                    
                    if rgb_val:
                        # Normalize RGB
                        current_rgb_norm = np.array([[[c / 255.0 for c in rgb_val]]])
                        current_lab = color.rgb2lab(current_rgb_norm)[0][0]
                        
                        # Find nearest neighbor using KDTree
                        dist, nearest_idx = tree.query(current_lab, k=1)
                        
                        nearest_name, _ = color_ref_metadata[nearest_idx]
                        
                        # Use actual hex code from the source RGB
                        actual_hex = rgb_to_hex(rgb_val)
                        formatted_palette.append(f"{nearest_name}({actual_hex})")
            
            if len(formatted_palette) != 5:
                continue
            
            doc_structure = {
                "description": ai_description,
                "palette": formatted_palette
            }
            
            documents.append(json.dumps(doc_structure))
            
            # Metadata must be primitives
            metadatas.append({
                "description": ai_description,
                "palette_str": ", ".join(formatted_palette)
            })
             
            ids.append(str(idx))
            
            if len(documents) >= batch_size:
                descriptions = [m["description"] for m in metadatas]
                batch_embeddings = ef(descriptions)

                collection.add(
                    ids=ids,
                    embeddings=batch_embeddings,
                    documents=documents,
                    metadatas=metadatas
                )

                documents.clear()
                metadatas.clear()
                ids.clear()

        except Exception as e:
            print(f"Error processing entry {idx}: {e}")
            continue

    if documents:
        descriptions = [m['description'] for m in metadatas]
        batch_embeddings = ef(descriptions)
        collection.add(
            ids=ids,
            embeddings=batch_embeddings,
            documents=documents,
            metadatas=metadatas
        )

    print(f"Database creation complete. Total items: {collection.count()}")

if __name__ == "__main__":
    main()
