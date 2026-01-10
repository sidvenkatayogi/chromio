import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from tqdm import tqdm
from skimage import color
import json
import pickle


# Configuration
CHROMA_PATH = "chroma_db_hsl"
COLLECTION_NAME = "pat"
MODEL_NAME = "all-mpnet-base-v2"

KD_TREE_PATH = "color_kdtree.pkl"

batch_size = 128

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def rgb_to_hsl(rgb):
    """
    Convert RGB tuple (0-255) to HSL format string (H, S%, L%).
    H is in [0, 360), S and L are in [0, 100] without decimals.
    
    Args:
        rgb: Tuple of (R, G, B) values in range 0-255
        
    Returns:
        String in format "(H, S%, L%)"
    """
    # Normalize RGB values to 0-1 range
    r, g, b = [x / 255.0 for x in rgb]
    
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # Calculate Lightness
    l = (max_val + min_val) / 2.0
    
    # Calculate Saturation
    if diff == 0:
        h = 0
        s = 0
    else:
        s = diff / (2.0 - max_val - min_val) if l > 0.5 else diff / (max_val + min_val)
        
        # Calculate Hue
        if max_val == r:
            h = ((g - b) / diff + (6 if g < b else 0)) / 6.0
        elif max_val == g:
            h = ((b - r) / diff + 2) / 6.0
        else:
            h = ((r - g) / diff + 4) / 6.0
    
    # Convert to proper ranges: H in [0, 360), S and L in [0, 100]
    h = int(h * 360) % 360
    s = int(s * 100)
    l = int(l * 100)
    
    return f"({h}, {s}%, {l}%)"

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

    print("Processing documents and populating database with HSL colors...")
    
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

                    # Convert RGB to HSL format
                    hsl_value = rgb_to_hsl(rgb_val)
                    formatted_palette.append(f"{nearest_name}{hsl_value}")
            
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
