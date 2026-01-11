import logging
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """
        Singleton manager for ChromaDB client and collection caching
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._client = None
            self._initialized = True
            self._collections_cache = {}
            logger.info("ChromaDBManager initialized")
    
    
    def get_client(self, db_path: str):
        """
            Get or create ChromaDB client with caching.
        """
        if self._client is None:
            logger.info(f"Creating new ChromaDB client at {db_path}")
            try:
                self._client = chromadb.PersistentClient(path=db_path)
            except Exception as e:
                logger.error(f"Error getting client from '{db_path}': {e}.\t Did you forget to initialize a chromadb instance in {db_path}?")
                raise
        return self._client
    
    
    def get_collection(self, db_path: str, collection_name: str = "pat", ef_model_name:str = "all-mpnet-base-v2"):
        """
            Get collection from cached client
        """
        if collection_name not in self._collections_cache:
            client = self.get_client(db_path)
            
            try:
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=ef_model_name)
                collection = client.get_collection(
                    name=collection_name,
                    embedding_function=ef
                )
                
                self._collections_cache[collection_name] = collection
                logger.info(f"Collection '{collection_name}' cached with embedding function")
            except Exception as e:
                logger.error(f"Error getting collection '{collection_name}': {e}")
                raise

        return self._collections_cache[collection_name]
    
    
    def close(self):
        """
            Close client and clear cache
            Must be done before initializing a new client
        """
        self._client = None
        self._collections_cache.clear()
        logger.info("ChromaDB client closed and collections cache cleared")
    
    
    ### color util extensions
    
    # sourced from evaluation_hsl.py
    def hsl_to_hex(self, hsl_str: str) -> str:
        """
            Convert HSL string format "(H, S%, L%)" to hex color "#RRGGBB".
        """
        # Parse HSL values
        match = re.match(r'\(\s*(\d{1,3})\s*,\s*(\d{1,3})%\s*,\s*(\d{1,3})%\s*\)', hsl_str)
        if not match:
            return "#000000"
        
        h, s, l = int(match.group(1)), int(match.group(2)), int(match.group(3))
        
        # Convert to 0-1 range
        h = h / 360.0
        s = s / 100.0
        l = l / 100.0
        
        # HSL to RGB conversion
        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p
        
        if s == 0:
            r = g = b = l
        else:
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)
        
        # Convert to 0-255 range and format as hex
        r_int = int(round(r * 255))
        g_int = int(round(g * 255))
        b_int = int(round(b * 255))
        
        return f"#{r_int:02x}{g_int:02x}{b_int:02x}"
    
    
    def hsl_list_to_hex_list(self, hsl_list: list[str]) -> list[str]:
        """
            Convert HSL string list to Hex string list
        """
        hex_list = [ self.hsl_to_hex(hsl_str) for hsl_str in hsl_list ]
        return hex_list