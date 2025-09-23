import argparse
import logging
import os

logger = logging.getLogger(__name__)

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import chromadb
COLLECTION_NAME = "nova_speedup_collection"

ROOT_DIR = "saves/chromadb"
client = chromadb.PersistentClient(path=ROOT_DIR)

def get_collection() -> chromadb.Collection:
    """
    Retrieves or creates the ChromaDB collection.
    """
    embedding_fct = DefaultEmbeddingFunction()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fct,  # type: ignore
    )

def delete(
    base_model: str,
    noise_epochs: int,
    noise_lr: float,
    reg_term: float,
    soft_target: bool,
) -> bool:
    """
    Deletes a document from the ChromaDB collection based on a sample vector and its metadata.
    """
    collection = get_collection()
    
    metadata_filter = {
            "$and": [
                {"base_model": {"$eq": base_model}},
                {"noise_epochs": {"$eq": noise_epochs}},
                {"noise_lr": {"$eq": noise_lr}},
                {"reg_term": {"$eq": reg_term}},
                {"soft_target": {"$eq": soft_target}},
            ]
        }

    try:
        results = collection.get(
            where=metadata_filter
        )

        if not results['ids']:
            print("❌ Document not found.")
            return False
        
        # Get all IDs before deleting to remove corresponding files
        ids_to_delete = results['ids']

        collection.delete(
            ids=ids_to_delete
        )

        # Now, delete the saved files
        for doc_id in ids_to_delete:
            anti_file = os.path.join(ROOT_DIR, "anti", f"{doc_id}.pt")
            prior_file = os.path.join(ROOT_DIR, "prior_embedding", f"{doc_id}.pt")
            
            if os.path.exists(anti_file):
                os.remove(anti_file)
            if os.path.exists(prior_file):
                os.remove(prior_file)
        
        print("✅ Successfully deleted document(s).")
        return True
    except Exception as e:
        print(f"❌ Failed to delete document(s): {e}")
        return False
    
def check_document_count():
    """
    Checks and logs the number of documents in the ChromaDB collection.
    """
    try:
        collection = get_collection()
        doc_count = collection.count()
        logger.info(f"The collection '{COLLECTION_NAME}' contains {doc_count} documents.")
        return doc_count
    except Exception as e:
        logger.error(f"Failed to count documents: {e}")
        return -1

def main():
    parser = argparse.ArgumentParser(description="Deletes ChromaDB entries for a specific training run.")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name.")
    parser.add_argument("--noise_epochs", type=int, required=True, help="Number of noise epochs.")
    parser.add_argument("--noise_lr", type=float, required=True, help="Noise learning rate.")
    parser.add_argument("--reg_term", type=float, required=True, help="Regularization term.")
    parser.add_argument("--soft_target", type=lambda x: x.lower() == 'true', required=True, help="Soft target flag.")
    
    args = parser.parse_args()

    delete(
        base_model=args.base_model,
        noise_epochs=args.noise_epochs,
        noise_lr=args.noise_lr,
        reg_term=args.reg_term,
        soft_target=args.soft_target
    )

    logger.info(f"The Count of the documents is: {check_document_count()}")

if __name__ == "__main__":
    main()