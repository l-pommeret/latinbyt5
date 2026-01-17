import argparse
from datasets import load_dataset
import os

def prepare_data(output_dir, subset_size=None):
    print(f"Loading PleIAs/Latin-PD dataset...")
    # Direct parquet loading to bypass metadata schema mismatch
    # The dataset metadata claims 'identifier' but files have 'file_id'
    dataset = load_dataset("parquet", data_files="https://huggingface.co/datasets/PleIAs/Latin-PD/resolve/main/data/*.parquet", split="train", streaming=True)
    
    print("Dataset loaded. Inspecting first few examples...")
    # Peek at first few examples
    for i, example in enumerate(dataset.take(3)):
        print(f"\n--- Example {i+1} ---")
        # Adjust key based on actual dataset structure, usually 'text' or 'content'
        text = example.get('text', example.get('content', str(example)))
        print(text[:200] + "...")

    output_path = os.path.join(output_dir, "train.txt")
    print(f"\nSaving data to {output_path}...")
    
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example.get('text', example.get('content', ''))
            if not text:
                continue
            
            # Simple cleaning if necessary, Byt5 is robust so we keep mostly raw
            f.write(text + "\n")
            
            count += 1
            if subset_size and count >= subset_size:
                break
            
            if count % 10000 == 0:
                print(f"Processed {count} documents...", end='\r')

    print(f"\nFinished processing {count} documents.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save processed data")
    parser.add_argument("--subset_size", type=int, default=10000, help="Number of documents to download for local training (default 10k to save space)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    prepare_data(args.output_dir, args.subset_size)
