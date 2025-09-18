import os
os.environ["CHROMA_TELEMETRY_IS_DISABLED"] = "1"

from torch import Tensor, tensor
from torch import (
    float16, float32, float64, int8, int16, int32, int64
)
from transformers import AutoTokenizer
import chromadb
import uuid
from typing import Dict, Any

# Initialize the ChromaDB client
ROOT_DIR = "saves/chromadb"
client = chromadb.PersistentClient(path=ROOT_DIR)

HF_TOKEN = "" # Your Token
TOKENIZER_MAPPING = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
}
# Define a constant for the collection name
COLLECTION_NAME = "nova_speedup_collection"

def get_collection() -> chromadb.Collection:
    """
    Retrieves or creates the ChromaDB collection.
    """

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

def get_tokenizer(
        base_model: str,
):
    """
    Create the Tokenizer for the parse model
    """
    model_path = None
    for name, path in TOKENIZER_MAPPING.items():
        if name in base_model:
            model_path = path
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=HF_TOKEN
    )
    # [meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.2-3B-Instruct, meta-llama/Llama-3.2-1B-Instruct]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 

    return tokenizer

def get_metadata(
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
    sample_key_str: str,
    anti_pattern_str: str = "",
    anti_pattern_dtype_str: str = "",
    as_filter: bool = False,
) -> Dict[str, Any]:
    """
    Creates a metadata dictionary from the input parameters.
    """
    if as_filter:
        return {
            "$and": [
                {"base_model": {"$eq": base_model}},
                {"noise_epochs": {"$eq": noise_epochs}},
                {"noise_lr": {"$eq": noise_lr}},
                {"reg_term": {"$eq": reg_term}},
                {"soft_target": {"$eq": soft_target}},
                {"sample_key_str": {"$eq": sample_key_str}},
            ]
        }
    if anti_pattern_str == "" and anti_pattern_dtype_str == "":
        raise Exception("No values parsed for: 'tensor_value', 'tensor_dtype'!")
    return {
        "base_model": base_model,
        "noise_epochs": noise_epochs,
        "noise_lr": noise_lr,
        "reg_term": reg_term,
        "soft_target": soft_target,
        "sample_key_str": sample_key_str,
        "anti_pattern_str": anti_pattern_str,
        "anti_pattern_dtype_str": anti_pattern_dtype_str,
    }

def put(
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
    sample: Tensor,
    anti_pattern: Tensor,
    embedding: Tensor,
):
    """
    Adds a key-value pair to the ChromaDB collection.
    """
    collection = get_collection()
    tokenizer = get_tokenizer(base_model=base_model)
    
    # Store the tensor value as a list and its datatype as a string in the metadata
    metadata = get_metadata(
        base_model, noise_epochs, noise_lr, reg_term, soft_target, str(sample.tolist()), str(anti_pattern.tolist()), str(anti_pattern.dtype),
    )
    
    # ChromaDB expects lists of values
    embedding_list = [embedding.tolist()]
    documents = [tokenizer.decode(sample.tolist(), skip_special_tokens=True)]
    metadatas = [metadata]
    ids = str(uuid.uuid4())

    collection.add(
        embeddings=embedding_list,
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"✅ Saved tensor mapping to ChromaDB collection: {COLLECTION_NAME}")

def get(
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
    sample: Tensor,
) -> Tensor:
    """
    Retrieves the value associated with a sample vector from ChromaDB.
    """
    collection = get_collection()

    # Define the filter for metadata, excluding the tensor value and dtype
    metadata_filter = get_metadata(
        base_model=base_model,
        noise_epochs=noise_epochs,
        noise_lr=noise_lr,
        reg_term=reg_term,
        soft_target=soft_target,
        sample_key_str=str(sample.tolist()),
        as_filter=True,
    )

    results = collection.get(
        where=metadata_filter
    )
    
    if results['metadatas'] and results['metadatas'][0]:
        # Retrieve the list and datatype from metadata and convert back to a tensor
        anti_pattern_str = results['metadatas'][0].get('anti_pattern_str')
        anti_pattern_dtype_str = results['metadatas'][0].get('anti_pattern_dtype_str')
        
        # Use a mapping to get the correct torch.dtype from the string
        # This is a robust way to handle the conversion
        dtype_map = {
            'torch.float16': float16,
            'torch.float32': float32,
            'torch.float64': float64,
            'torch.int8':  int8,
            'torch.int16': int16,
            'torch.int32': int32,
            'torch.int64': int64,
        }
        
        dtype = dtype_map.get(anti_pattern_dtype_str, None)
        if dtype is None:
            raise ValueError(f"Unknown tensor dtype: {anti_pattern_dtype_str}")
        return tensor(eval(anti_pattern_str), dtype=dtype)
    else:
        raise KeyError("The sample you are looking for does not exist")

def exists(
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
    sample: Tensor,
) -> bool:
    """
    Checks if a sample vector and its associated metadata exist in ChromaDB.
    """
    collection = get_collection()

    metadata_filter = get_metadata(
        base_model=base_model,
        noise_epochs=noise_epochs,
        noise_lr=noise_lr,
        reg_term=reg_term,
        soft_target=soft_target,
        sample_key_str=str(sample.tolist()),
        as_filter=True,
    )
    
    results = collection.get(
        where=metadata_filter
    )
    
    return bool(results['ids'])

def delete(
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
    sample: Tensor,
) -> bool:
    """
    Deletes a document from the ChromaDB collection based on a sample vector and its metadata.
    """
    collection = get_collection()
    
    metadata_filter = get_metadata(
        base_model=base_model,
        noise_epochs=noise_epochs,
        noise_lr=noise_lr,
        reg_term=reg_term,
        soft_target=soft_target,
        sample_key_str=str(sample.tolist()),
        as_filter=True,
    )

    try:
        collection.delete(
            where=metadata_filter
        )
        print("✅ Successfully deleted document.")
        return True
    except Exception as e:
        print(f"❌ Failed to delete document: {e}")
        return False