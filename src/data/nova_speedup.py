import os
import json
from torch import Tensor

ROOT_DIR = "saves/nova_speedup"

def load_json(file_path: str):
    """
    Loads a JSON file and returns the data.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path: str):
    """
    Saves a dictionary to a JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def get_relative_paths():
    """
    Returns a set of relative paths for all files in a given directory and its subdirectories.
    """
    if not os.path.exists(ROOT_DIR):
        return set()
    relative_paths = set()
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        for filename in filenames:
            absolute_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(absolute_path, ROOT_DIR)
            relative_paths.add(relative_path)
    return relative_paths

def get_query( 
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
) -> str:
    config_list = [base_model, noise_epochs, noise_lr, reg_term, soft_target]
    config_list = [str(var).replace(".", "_") for var in config_list]
    config_list = os.path.join(*config_list)
    return config_list + ".json"

def create_directory(
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
):
    query = get_query(base_model=base_model, noise_epochs=noise_epochs, noise_lr=noise_lr, reg_term=reg_term, soft_target=soft_target,)
    relative_path = query.rsplit(os.sep, maxsplit=1)[0]
    full_path = os.path.join(ROOT_DIR, relative_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"✅ Created directory: {full_path}")
    else:
        print(f"ℹ️ Directory already exists: {full_path}")

def put(
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
    key: Tensor,
    value: Tensor,
):
    create_directory(
        base_model=base_model, noise_epochs=noise_epochs, noise_lr=noise_lr, reg_term=reg_term, soft_target=soft_target,
    )
    query = os.path.join(ROOT_DIR, get_query(base_model=base_model, noise_epochs=noise_epochs, noise_lr=noise_lr, reg_term=reg_term, soft_target=soft_target, ))
    
    # Load JSON file if it exists, otherwise start with an empty dictionary
    dictionary = load_json(query) if os.path.exists(query) else {}

    # Convert the key tensor and value tensor to a JSON-compatible format (list)
    hashable_tensor = key.tolist()
    dictionary[str(hashable_tensor)] = value.tolist()  # Store value as a list

    # Save the updated dictionary as a JSON file
    save_json(dictionary, query)
    print(f"✅ Saved tensor mapping to: {query}")

def get(
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
    sample: Tensor,
) -> Tensor:
    query = os.path.join(ROOT_DIR, get_query(base_model=base_model, noise_epochs=noise_epochs, noise_lr=noise_lr, reg_term=reg_term, soft_target=soft_target, ))
    dictionary = load_json(query)

    try:
        hashable_tensor_str = str(sample.tolist())
        mapping = dictionary[hashable_tensor_str]
        return Tensor(mapping)  # Convert the list back to a Tensor
    except KeyError:
        raise KeyError("The sample you are looking for does not exist")

def exists(
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
    sample: Tensor,
) -> bool:
    query = get_query(base_model=base_model, noise_epochs=noise_epochs, noise_lr=noise_lr, reg_term=reg_term, soft_target=soft_target,)
    rel_path_set = get_relative_paths()
    if query in rel_path_set:
        full_query_path = os.path.join(ROOT_DIR, query)
        loaded_dict = load_json(full_query_path)
       
        hashable_tensor_str = str(sample.tolist())
        if hashable_tensor_str in loaded_dict:
            return True
    return False