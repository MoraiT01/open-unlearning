import os
os.environ["CHROMA_TELEMETRY_IS_DISABLED"] = "1"

from torch import Tensor, tensor, flip, nonzero
from torch import (
    float16, float32, float64, int8, int16, int32, int64
)
from transformers import AutoTokenizer
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import chromadb
import uuid
from typing import Dict, Any

# Initialize the ChromaDB client
ROOT_DIR = "saves/chromadb"
client = chromadb.PersistentClient(path=ROOT_DIR)

HF_TOKEN = # Your Token
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
    
    embedding_fct = DefaultEmbeddingFunction()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fct,
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

def reduce_eos_tokens(
        vector: Tensor
) -> Tensor:
    """
    Removes trailing duplicate values from a 1D PyTorch Tensor.

    Args:
        tensor (torch.Tensor): A 1D tensor.

    Returns:
        torch.Tensor: A new tensor with trailing duplicates removed.
    """
    if vector.dim() != 1 or vector.numel() == 0:
        return vector
    # Get the last value of the vector
    last_value = vector[-1]

    # Reverse the vector to find the first element that's different
    reversed_vector = flip(vector, dims=[0])

    # Find all indices where the values are NOT equal to the last value
    diff_indices = nonzero(reversed_vector != last_value)
    if diff_indices.numel() == 0:
        # If no different values are found, the entire vector is the tail
        cutter = len(vector)
    else:
        # The length of the tail is the index of the first different value
        cutter =  diff_indices[0].item()

    if cutter == 1:
        return vector
    return vector[:-(cutter-1)]

def get_metadata(
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
    tensor_key: str,
    tensor_value: str = None,
    tensor_dtype: str = None,
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
                {"tensor_key": {"$eq": tensor_key}},
            ]
        }
    if tensor_dtype == None and tensor_value == None:
        raise Exception("No values parsed for: 'tensor_value', 'tensor_dtype'!")
    return {
        "base_model": base_model,
        "noise_epochs": noise_epochs,
        "noise_lr": noise_lr,
        "reg_term": reg_term,
        "soft_target": soft_target,
        "tensor_key": tensor_key,
        "tensor_value": tensor_value,
        "tensor_dtype": tensor_dtype,
    }

def put(
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
    key: Tensor,
    value: Tensor,
):
    """
    Adds a key-value pair to the ChromaDB collection.
    """
    collection = get_collection()
    tokenizer = get_tokenizer(base_model=base_model)
    key = reduce_eos_tokens(key)
    
    # Store the tensor value as a list and its datatype as a string in the metadata
    metadata = get_metadata(
        base_model, noise_epochs, noise_lr, reg_term, soft_target, str(key.tolist()), str(value.tolist()), str(value.dtype),
    )
    
    # ChromaDB expects lists of values
    embeddings = None 
    documents = [tokenizer.decode(key)]    # Could add actual decoded text
    metadatas = [metadata]
    ids = str(uuid.uuid4())

    collection.add(
        embeddings=embeddings,
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
    
    sample = reduce_eos_tokens(sample)
    # Define the filter for metadata, excluding the tensor value and dtype
    metadata_filter = get_metadata(
        base_model=base_model,
        noise_epochs=noise_epochs,
        noise_lr=noise_lr,
        reg_term=reg_term,
        soft_target=soft_target,
        tensor_key=str(sample.tolist()),
        as_filter=True,
    )

    results = collection.get(
        where=metadata_filter
    )
    
    if results['metadatas'] and results['metadatas'][0]:
        # Retrieve the list and datatype from metadata and convert back to a tensor
        tensor_str = results['metadatas'][0].get('tensor_value')
        tensor_dtype_str = results['metadatas'][0].get('tensor_dtype')
        
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
        
        dtype = dtype_map.get(tensor_dtype_str, None)
        if dtype is None:
            raise ValueError(f"Unknown tensor dtype: {tensor_dtype_str}")
        return tensor(eval(tensor_str), dtype=dtype)
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

    sample = reduce_eos_tokens(sample)
    metadata_filter = get_metadata(
        base_model=base_model,
        noise_epochs=noise_epochs,
        noise_lr=noise_lr,
        reg_term=reg_term,
        soft_target=soft_target,
        tensor_key=str(sample.tolist()),
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
    
    sample = reduce_eos_tokens(sample)
    metadata_filter = get_metadata(
        base_model=base_model,
        noise_epochs=noise_epochs,
        noise_lr=noise_lr,
        reg_term=reg_term,
        soft_target=soft_target,
        tensor_key=str(sample.tolist()),
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