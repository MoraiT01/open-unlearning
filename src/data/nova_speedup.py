import os
import shutil
import torch
from torch import Tensor, load, save

ROOT_DIR = "../saves/nove_speedup"

# --- Your functions to be tested ---
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
    return config_list + ".pkl"

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
    dictionary = load(query) if os.path.exists(query) else {}
    dictionary[tuple(key.tolist())] = value
    save(dictionary, query)
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
    dictionary = load(query)
    try:
        mapping = dictionary[tuple(sample.tolist())]
    except KeyError:
        raise KeyError("The sample you are looking for does not exist")
    return mapping

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
        loaded_dict = load(full_query_path)
        if tuple(sample.tolist()) in loaded_dict:
            return True
    return False

# --- Main Test Execution ---
if __name__ == "__main__":
    print("--- Starting Comprehensive Cache Test ---")
    
    # --- Cleanup from previous runs ---
    if os.path.exists(ROOT_DIR):
        shutil.rmtree(ROOT_DIR)
        print(f"🗑️ Cleaned up previous test directory: {ROOT_DIR}")

    # --- Test Case 1: Saving and retrieving a single entry ---
    print("\n--- Test Case 1: Basic Save and Retrieve ---")
    
    # Parameters
    params1 = {
        'base_model': 'resnet18',
        'noise_epochs': 5,
        'noise_lr': 0.001,
        'reg_term': 0.1,
        'soft_target': True
    }
    key_tensor1 = torch.tensor([1, 2, 3])
    value_tensor1 = torch.tensor([10, 20, 30])
    
    # Save the mapping
    put(**params1, key=key_tensor1, value=value_tensor1)
    
    # Check if the entry exists
    assert exists(**params1, sample=key_tensor1), "Test 1 Failed: 'exists' returned False for a known key."
    print("✅ Success: `exists` correctly found the key.")
    
    # Retrieve the value
    retrieved_value1 = get(**params1, sample=key_tensor1)
    assert torch.equal(retrieved_value1, value_tensor1), "Test 1 Failed: Retrieved value does not match original."
    print("✅ Success: `get` correctly retrieved the value.")
    
    # --- Test Case 2: Adding a new entry to an existing file ---
    print("\n--- Test Case 2: Adding to an Existing File ---")
    
    key_tensor2 = torch.tensor([4, 5, 6])
    value_tensor2 = torch.tensor([40, 50, 60])
    
    put(**params1, key=key_tensor2, value=value_tensor2)
    
    # Check for both original and new entries
    assert exists(**params1, sample=key_tensor1), "Test 2 Failed: Original key was lost."
    assert exists(**params1, sample=key_tensor2), "Test 2 Failed: New key was not added."
    print("✅ Success: Both original and new keys were found after addition.")

    # --- Test Case 3: Handling non-existent keys and files ---
    print("\n--- Test Case 3: Handling Non-Existent Keys and Files ---")
    
    non_existent_key = torch.tensor([99, 98, 97])
    
    # Check `exists` for a non-existent key
    assert not exists(**params1, sample=non_existent_key), "Test 3 Failed: 'exists' returned True for a non-existent key."
    print("✅ Success: `exists` correctly handled a non-existent key.")
    
    # Test `get` for a non-existent key
    try:
        get(**params1, sample=non_existent_key)
        assert False, "Test 3 Failed: 'get' did not raise KeyError for a non-existent key."
    except KeyError:
        print("✅ Success: `get` correctly raised KeyError.")

    # Test with a completely different set of parameters (a new file)
    new_params = {
        'base_model': 'vgg16',
        'noise_epochs': 2,
        'noise_lr': 0.05,
        'reg_term': 0.01,
        'soft_target': False
    }
    try:
        get(**new_params, sample=key_tensor1)
        assert False, "Test 3 Failed: 'get' did not raise FileNotFoundError for a new query."
    except FileNotFoundError:
        print("✅ Success: `get` correctly raised FileNotFoundError for a new query.")

    # --- Test Case 4: Testing get_relative_paths function ---
    print("\n--- Test Case 4: Testing get_relative_paths ---")
    
    # Add another file to the directory
    put(**new_params, key=key_tensor1, value=value_tensor1)
    
    # Get the set of paths and check for existence of both files
    path_set = get_relative_paths()
    expected_path1 = get_query(**params1)
    expected_path2 = get_query(**new_params)
    
    assert expected_path1 in path_set, "Test 4 Failed: get_relative_paths missing first file."
    assert expected_path2 in path_set, "Test 4 Failed: get_relative_paths missing second file."
    print("✅ Success: `get_relative_paths` correctly identified both files.")

    # # --- Final cleanup ---
    # print("\n--- Final cleanup ---")
    # if os.path.exists(ROOT_DIR):
    #     shutil.rmtree(ROOT_DIR)
    #     print(f"🗑️ Test directory has been removed: {ROOT_DIR}")

    print("\n--- All tests passed! ---")