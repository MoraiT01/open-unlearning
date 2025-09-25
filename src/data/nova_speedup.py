import os
import shutil
import uuid
from typing import Dict, Any

import logging
from contextlib import contextmanager

from torch import Tensor, save, load
from torch import (
    float16, float32, float64, int8, int16, int32, int64
)
from transformers import AutoTokenizer
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import chromadb

# Initialize logging
logger = logging.getLogger(__name__)

# Constants
HF_TOKEN = "" # Your HuggingFace Token
TOKENIZER_MAPPING = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
}

# The ROOT_DIR for temporary storage, managed by the context manager
BASE_ROOT_DIR = "saves/chromadb"
COLLECTION_NAME = "nova_speedup_collection"

# -----------------------------------------------------------------------------
# Context Manager Class for Ephemeral Database
# -----------------------------------------------------------------------------
class DatabaseManager:
    """
    A context manager to handle an ephemeral ChromaDB session, including 
    the creation and cleanup of temporary directories for tensor storage.
    """
    def __init__(self):
        self.temp_dir = os.path.join(BASE_ROOT_DIR, str(uuid.uuid4()))
        self.client = chromadb.PersistentClient(path=self.temp_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=DefaultEmbeddingFunction(), # type: ignore
        )

    def __enter__(self):
        """
        Set up the temporary database and file directories.
        """
        os.makedirs(os.path.join(self.temp_dir, "anti"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "prior_embedding"), exist_ok=True)
        logger.info(f"Ephemeral ChromaDB session started at: {self.temp_dir}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Clean up all temporary files and the database directory.
        """
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            logger.info(f"✅ Successfully cleaned up ephemeral session: {self.temp_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to clean up temporary directory: {e}")

# -----------------------------------------------------------------------------
# Helper Functions (refactored to accept client and collection)
# -----------------------------------------------------------------------------
def get_tokenizer(base_model: str):
    """Create the Tokenizer for the parsed model."""
    model_path = TOKENIZER_MAPPING.get(base_model)
    if not model_path:
        raise ValueError(f"Model {base_model} not found in tokenizer mapping.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def get_metadata(
    base_model: str, noise_epochs: int, noise_lr: float, reg_term: float, 
    soft_target: bool, sample_key_str: str = None, 
    anti_pattern_dtype_str: str = "", as_filter: bool = False,
) -> Dict[str, Any]:
    """Creates a metadata dictionary from the input parameters."""
    # (Same logic as before, but with more explicit checks)
    if as_filter:
        metadata_filter = {
            "base_model": {"$eq": base_model},
            "noise_epochs": {"$eq": noise_epochs},
            "noise_lr": {"$eq": noise_lr},
            "reg_term": {"$eq": reg_term},
            "soft_target": {"$eq": soft_target},
        }
        if sample_key_str is not None:
            metadata_filter["sample_key_str"] = {"$eq": sample_key_str}
        return {"$and": [metadata_filter]}
    
    if not anti_pattern_dtype_str:
        raise ValueError("No values parsed for 'anti_pattern_dtype_str'!")
        
    return {
        "base_model": base_model,
        "noise_epochs": noise_epochs,
        "noise_lr": noise_lr,
        "reg_term": reg_term,
        "soft_target": soft_target,
        "sample_key_str": sample_key_str,
        "anti_pattern_dtype_str": anti_pattern_dtype_str,
    }

# -----------------------------------------------------------------------------
# CRUD Operations (refactored to use the DatabaseManager context)
# -----------------------------------------------------------------------------

def put(
    db_manager: DatabaseManager,
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
    sample: Tensor,
    anti_pattern: Tensor,
    sample_embedding: Tensor,
):
    """Adds a key-value pair to the ChromaDB collection."""
    tokenizer = get_tokenizer(base_model=base_model)
    ids = str(uuid.uuid4())
    
    metadata = get_metadata(
        base_model, noise_epochs, noise_lr, reg_term, soft_target,
        sample_key_str=str(sample.tolist()),
        anti_pattern_dtype_str=str(anti_pattern.dtype),
    )
    
    documents = [tokenizer.decode(sample.tolist(), skip_special_tokens=True)]
    
    db_manager.collection.add(
        embeddings=None, # The default embedding function will handle this
        documents=documents,
        metadatas=[metadata], # type: ignore
        ids=[ids],
    )
    
    try:
        save(anti_pattern, os.path.join(db_manager.temp_dir, "anti", f"{ids}.pt"))
        save(sample_embedding, os.path.join(db_manager.temp_dir, "prior_embedding", f"{ids}.pt"))
        logger.info(f"✅ Saved tensors to temporary directory for ID: {ids}")
    except Exception as e:
        logger.error(f"Failed to save tensor: {e}")

    logger.info(f"✅ Saved tensor mapping to ChromaDB collection: {COLLECTION_NAME}")

def get(
    db_manager: DatabaseManager,
    base_model: str, noise_epochs: int, noise_lr: float, reg_term: float, 
    soft_target: bool, sample: Tensor, to: str,
) -> Tensor:
    """Retrieves the value associated with a sample vector from ChromaDB."""
    metadata_filter = get_metadata(
        base_model=base_model, noise_epochs=noise_epochs, noise_lr=noise_lr, 
        reg_term=reg_term, soft_target=soft_target, 
        sample_key_str=str(sample.tolist()), as_filter=True,
    )
    
    results = db_manager.collection.get(where=metadata_filter)
    
    if results['metadatas'] and results['metadatas'][0]:
        doc_id = results['ids'][0]
        anti_patter_tensor = load(
            os.path.join(db_manager.temp_dir, "anti", f"{doc_id}.pt"), 
            map_location=to
        )
        anti_pattern_dtype_str = results['metadatas'][0].get('anti_pattern_dtype_str')
        
        dtype_map = {
            'torch.float16': float16, 'torch.float32': float32, 'torch.float64': float64,
            'torch.int8': int8, 'torch.int16': int16, 'torch.int32': int32, 'torch.int64': int64,
        }
        
        dtype = dtype_map.get(anti_pattern_dtype_str, None) # type: ignore
        if dtype is None:
            raise ValueError(f"Unknown tensor dtype: {anti_pattern_dtype_str}")
        
        return anti_patter_tensor
    else:
        raise KeyError("The sample you are looking for does not exist")

def exists(
    db_manager: DatabaseManager,
    base_model: str, noise_epochs: int, noise_lr: float, reg_term: float, 
    soft_target: bool, sample: Tensor,
) -> bool:
    """Checks if a sample vector and its associated metadata exist in ChromaDB."""
    metadata_filter = get_metadata(
        base_model=base_model, noise_epochs=noise_epochs, noise_lr=noise_lr,
        reg_term=reg_term, soft_target=soft_target,
        sample_key_str=str(sample.tolist()), as_filter=True,
    )
    
    results = db_manager.collection.get(where=metadata_filter)
    return bool(results['ids'])

def delete(
    db_manager: DatabaseManager,
    base_model: str, noise_epochs: int, noise_lr: float, reg_term: float, 
    soft_target: bool, sample: Tensor = None,
) -> bool:
    """Deletes a document and its corresponding tensors from the ephemeral session."""
    metadata_filter = get_metadata(
        base_model=base_model, noise_epochs=noise_epochs, noise_lr=noise_lr,
        reg_term=reg_term, soft_target=soft_target,
        sample_key_str=str(sample.tolist()) if sample is not None else None,
        as_filter=True,
    )
    
    results = db_manager.collection.get(where=metadata_filter)
    if not results['ids']:
        logger.warning("❌ Document not found for deletion.")
        return False
        
    ids_to_delete = results['ids']
    
    try:
        db_manager.collection.delete(ids=ids_to_delete)
        for doc_id in ids_to_delete:
            anti_file = os.path.join(db_manager.temp_dir, "anti", f"{doc_id}.pt")
            prior_file = os.path.join(db_manager.temp_dir, "prior_embedding", f"{doc_id}.pt")
            if os.path.exists(anti_file):
                os.remove(anti_file)
            if os.path.exists(prior_file):
                os.remove(prior_file)
        logger.info("✅ Successfully deleted document(s) and tensors.")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to delete document(s): {e}")
        return False

def check_document_count(db_manager: DatabaseManager):
    """Checks and logs the number of documents in the ChromaDB collection."""
    try:
        doc_count = db_manager.collection.count()
        logger.info(f"The collection '{COLLECTION_NAME}' contains {doc_count} documents.")
        return doc_count
    except Exception as e:
        logger.error(f"Failed to count documents: {e}")
        return -1