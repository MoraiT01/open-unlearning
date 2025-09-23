import argparse
from src.data.nova_speedup import delete, check_document_count # Make sure to import the correct function
import logging

logger = logging.getLogger(__name__)

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