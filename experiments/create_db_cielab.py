import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from tqdm import tqdm
from skimage import color
import json
import pickle


# Configuration
CHROMA_PATH = "chroma_db_cielab"
COLLECTION_NAME = "pat"
MODEL_NAME = "all-mpnet-base-v2"

KD_TREE_PATH = "color_kdtree.pkl"

batch_size = 128

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def rgb_to_lab(rgb):
    """
    Convert RGB tuple (0-255) to CIELAB format string L(L, a, b).
    L is in [0, 100], a and b are in [-128, 127], no decimals.
    
    Args:
        rgb: Tuple of (R, G, B) values in range 0-255
        
    Returns:
        String in format "L(L, a, b)"
    """
    # Normalize RGB to 0-1 range and convert to LAB
    rgb_normalized = np.array([[[c / 255.0 for c in rgb]]])
    lab = color.rgb2lab(rgb_normalized)[0][0]
    
    # Round to integers
    L = int(round(lab[0]))
    a = int(round(lab[1]))
    b = int(round(lab[2]))
    
    # Clamp values to valid ranges
    L = max(0, min(100, L))
    a = max(-128, min(127, a))
    b = max(-128, min(127, b))
    
    return f"L({L}, {a}, {b})"

def main():
    print("Loading prebuilt KDTree...")
    with open(KD_TREE_PATH, "rb") as f:
        data = pickle.load(f)

    tree = data["tree"]
    color_ref_metadata = data["metadata"]

    print("KDTree loaded successfully.")

    # Load hexcolor dataset
    print("Loading hexcolor dataset...")
    with open("hexcolor_vf/train_names.pkl", "rb") as f:
        train_names = pickle.load(f)
    with open("hexcolor_vf/train_palettes_rgb.pkl", "rb") as f:
        train_palettes = pickle.load(f)
    print(f"Loaded {len(train_names)} entries.")

    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Create embedding function
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    
    # Get or create collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

    print("Processing documents and populating database with CIELAB colors...")
    
    documents = []
    metadatas = []
    ids = []
    
    # Iterate with index to use as ID
    for idx, (name_tokens, palette_rgb) in enumerate(tqdm(zip(train_names, train_palettes), total=len(train_names))):

        try:
            desc = " ".join(name_tokens)
            if not desc:
                continue
                
            if not palette_rgb or len(palette_rgb) != 5:
                continue

            # Extract colors
            formatted_palette = []
            
            for rgb_val in palette_rgb:
                if rgb_val:
                    # Normalize RGB
                    current_rgb_norm = np.array([[[c / 255.0 for c in rgb_val]]])
                    current_lab = color.rgb2lab(current_rgb_norm)[0][0]
                    
                    # Find nearest neighbor using KDTree
                    dist, nearest_idx = tree.query(current_lab, k=1)

                    # Get names and combine them
                    nearest_name, _ = color_ref_metadata[nearest_idx]

                    # Convert RGB to CIELAB format
                    lab_value = rgb_to_lab(rgb_val)
                    formatted_palette.append(f"{nearest_name}{lab_value}")
            
            if len(formatted_palette) != 5:
                continue
            
            doc_structure = {
                "description": desc,
                "palette": formatted_palette
            }
            
            documents.append(json.dumps(doc_structure))
             
            ids.append(str(idx))
            
            if len(documents) >= batch_size:
                descriptions = [json.loads(d)["description"] for d in documents]
                batch_embeddings = ef(descriptions)

                collection.add(
                    ids=ids,
                    embeddings=batch_embeddings,
                    documents=documents
                )

                documents.clear()
                ids.clear()

        except Exception as e:
            print(f"Error processing entry {idx}: {e}")
            continue

    if documents:
        descriptions = [json.loads(d)['description'] for d in documents]
        batch_embeddings = ef(descriptions)
        collection.add(
            ids=ids,
            embeddings=batch_embeddings,
            documents=documents
        )

    print(f"Database creation complete. Total items: {collection.count()}")

if __name__ == "__main__":
    main()
