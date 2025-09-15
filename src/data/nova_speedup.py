from torch import Tensor, tensor, flip, nonzero
from torch import (
    float16, float32, float64, int8, int16, int32, int64
)
import chromadb
from typing import Dict, Any, List

# Initialize the ChromaDB client
ROOT_DIR = "saves/chromadb"
client = chromadb.PersistentClient(path=ROOT_DIR)

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
    tensor_value: List[float] | List[int] = None,
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
    
    # Store the tensor value as a list and its datatype as a string in the metadata
    metadata = get_metadata(
        base_model, noise_epochs, noise_lr, reg_term, soft_target, value.tolist(), str(value.dtype)
    )
    key = reduce_eos_tokens(key)
    
    # ChromaDB expects lists of values
    embeddings = [key.tolist()]
    documents = [""]
    metadatas = [metadata]
    ids = [str(hash(tuple(key.tolist())))]

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
    metadata_filter = {
        "base_model": base_model,
        "noise_epochs": noise_epochs,
        "noise_lr": noise_lr,
        "reg_term": reg_term,
        "soft_target": soft_target,
    }

    results = collection.get(
        ids=[str(hash(tuple(sample.tolist())))],
        where=metadata_filter
    )
    
    if results['metadatas'] and results['metadatas'][0]:
        # Retrieve the list and datatype from metadata and convert back to a tensor
        tensor_list = results['metadatas'][0].get('tensor_value')
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

        return tensor(tensor_list, dtype=dtype)
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
        as_filter=True,
    )
    
    results = collection.get(
        ids=[str(hash(tuple(sample.tolist())))],
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
        as_filter=True,
    )

    try:
        collection.delete(
            ids=[str(hash(tuple(sample.tolist())))],
            where=metadata_filter
        )
        print("✅ Successfully deleted document.")
        return True
    except Exception as e:
        print(f"❌ Failed to delete document: {e}")
        return False