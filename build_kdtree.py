import numpy as np
import pickle
from datasets import load_dataset
from tqdm import tqdm
from skimage import color

KD_TREE_PATH = "color_kdtree.pkl"

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def main():
    print("Loading color reference dataset...")
    color_dataset = load_dataset("boltuix/color-pedia", split="train")

    color_ref_lab = []
    color_ref_metadata = []  # (color_name, hex)

    print("Converting HEX → LAB...")
    for item in tqdm(color_dataset):
        try:
            hex_code = item["HEX Code"]
            name = item["Color Name"]

            rgb = hex_to_rgb(hex_code)
            rgb_norm = np.array([[[c / 255.0 for c in rgb]]])
            lab = color.rgb2lab(rgb_norm)[0][0]

            color_ref_lab.append(lab)
            color_ref_metadata.append((name, hex_code))
        except Exception:
            continue

    color_ref_lab = np.array(color_ref_lab, dtype=np.float32)

    if len(color_ref_lab) == 0:
        raise RuntimeError("No colors loaded; KDTree not built.")

    print("Building KDTree...")
    from scipy.spatial import cKDTree
    tree = cKDTree(color_ref_lab)

    print(f"Saving KDTree → {KD_TREE_PATH}")
    with open(KD_TREE_PATH, "wb") as f:
        pickle.dump(
            {
                "tree": tree,
                "metadata": color_ref_metadata,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print("KDTree build complete.")

if __name__ == "__main__":
    main()