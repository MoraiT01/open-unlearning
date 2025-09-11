import os
from torch import Tensor, load, save

ROOT_DIR = "saves/nova_speedup"

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
    dictionary = load(query, weights_only=True) if os.path.exists(query) else {}

    hashable_tensor = tuple(key.tolist())
    dictionary[hashable_tensor] = value
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
    dictionary = load(query, weights_only=True)
    try:
        hashable_tensor = tuple(sample.tolist())
        mapping = dictionary[hashable_tensor]
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
        loaded_dict = load(full_query_path, weights_only=True)
        hashable_tensor = tuple(sample.tolist())
        if hashable_tensor in loaded_dict:
            return True
    return False
