# parallel_tokenize.py (Updated with tokenizer padding)

import os
import pickle
import numpy as np
import glob
from tqdm import tqdm
import sys
import multiprocessing as mp
from functools import partial
import random
import tiktoken

# Config
shard_size = 10_000_000
data_dir = "data"
dataset_folder = "finewebedu10b/fineweb_chunks"
output_dir = "tokenized_data"
tiktoken_model_name = "cl100k_base"
max_workers = max(1, mp.cpu_count() // 2)
power_of_base_padding = 64  # Pad vocab to nearest power of this value

try:
    enc = tiktoken.get_encoding(tiktoken_model_name)
    print(f"Loaded tiktoken encoder '{tiktoken_model_name}' with {enc.n_vocab:,} tokens")
except Exception as e:
    print(f"Error initializing tiktoken encoder '{tiktoken_model_name}': {e}")
    sys.exit(1)

def n64(n):
    return (64*(n//64))+64

def write_shard(filename, tokens):
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()

    header = np.array([20240520, 1, len(tokens)], dtype=np.uint32) # Magic, Version, Length

    dtype = np.uint32

    try:
        with open(filename, 'wb') as f:
            f.write(header.tobytes())
            f.write(np.array(tokens, dtype=dtype).tobytes())
    except IOError as e:
        print(f"Error writing shard {filename}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error writing shard {filename}: {e}", file=sys.stderr)

worker_enc = None

def init_worker(model_name):
    global worker_enc
    try:
        worker_enc = tiktoken.get_encoding(model_name)
    except Exception as e:
        print(f"Error initializing tiktoken in worker {os.getpid()}: {e}", file=sys.stderr)
        raise e

def process_chunk(file_path):
    """Tokenizes a single file using the worker's tiktoken encoder.
       Always returns tuple (error_message_or_None, list_of_tokens)."""
    global worker_enc
    if worker_enc is None:
        # This should not happen if init_worker is called correctly, if you see this, you messed up
        return (f"Error: Tiktoken encoder not initialized in worker for file {file_path}", [])

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = worker_enc.encode_ordinary(text)
        return (None, tokens)
    except FileNotFoundError:
        return (f"Error: File not found {file_path}", [])
    except UnicodeDecodeError as e:
        return (f"Error decoding file {file_path}: {e}", [])
    except Exception as e:
        return (f"Error processing {file_path}: {e}", [])

def parallel_tokenize(files, workers, model_name):
    """mp pool"""
    results = []
    errors = []

    chunksize = max(1, len(files) // (workers * 4) if workers > 0 else len(files))

    try:
        with mp.Pool(workers, initializer=init_worker, initargs=(model_name,)) as pool:
            with tqdm(total=len(files), desc="Tokenizing Chunks") as pbar:
                for error_msg, tokens in pool.imap(process_chunk, files, chunksize=chunksize):
                    if error_msg:
                        errors.append(error_msg)
                    elif tokens: # Check if tokens list is not empty
                        results.extend(tokens)
                    pbar.update(1)

    except Exception as e:
        print(f"\nCritical error during parallel processing: {e}", file=sys.stderr) # kinda basic handling
        return [], errors

    if errors:
        print(f"\nEncountered {len(errors)} errors during tokenization (showing first 10):")
        for i, err in enumerate(errors[:10]):
            print(f"  {i+1}. {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")

    return results

def create_shards(tokens, output_dir, split_name):
    os.makedirs(output_dir, exist_ok=True)
    num_shards = (len(tokens) + shard_size - 1) // shard_size
    print(f"Writing {len(tokens):,} tokens into {num_shards} shards for '{split_name}' split...")

    shard_tokens = []

    token_count = 0
    shard_index = 0

    for token in tqdm(tokens, desc=f"Writing {split_name} shards", total=len(tokens)):
        shard_tokens.append(token)
        token_count += 1
        if len(shard_tokens) >= shard_size:
            shard_filename = os.path.join(output_dir, f"{split_name}_{shard_index:06d}.bin")
            write_shard(shard_filename, shard_tokens)
            shard_index += 1
            shard_tokens = []

    if shard_tokens:
        shard_filename = os.path.join(output_dir, f"{split_name}_{shard_index:06d}.bin")
        write_shard(shard_filename, shard_tokens)
        shard_index += 1 # match num_shards calculated earlier

    if shard_index != num_shards:
         print(f"Warning: Expected {num_shards} shards, but wrote {shard_index} for {split_name}", file=sys.stderr)

    return shard_index

if __name__ == '__main__':

    input_path = os.path.join(data_dir, dataset_folder)
    print(f"Searching for .txt files in: {input_path}")
    files = sorted(glob.glob(os.path.join(input_path, "*.txt")))[:100]

    if not files:
        print(f"Error: No .txt files found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} .txt files.")

    # Shuffle and split files
    random.seed(42)
    random.shuffle(files)
    split_idx = int(0.9 * len(files))
    train_files, val_files = files[:split_idx], files[split_idx:]
    print(f"Splitting into {len(train_files)} train files and {len(val_files)} validation files.")

    print(f"\nStarting tokenization for TRAINING split ({len(train_files)} files)...")
    train_tokens = parallel_tokenize(train_files, max_workers, tiktoken_model_name)
    if not train_tokens:
        print("No training tokens were generated. Check errors.", file=sys.stderr)
    else:
        print(f"Successfully tokenized training files. Total tokens: {len(train_tokens):,}")
        train_shards = create_shards(train_tokens, output_dir, "train")
        print(f"Finished writing {train_shards} training shards.")
    del train_tokens

    print(f"\nStarting tokenization for VALIDATION split ({len(val_files)} files)...")
    val_tokens = parallel_tokenize(val_files, max_workers, tiktoken_model_name)
    if not val_tokens:
        print("No validation tokens were generated. Check errors.", file=sys.stderr)
    else:
        print(f"Successfully tokenized validation files. Total tokens: {len(val_tokens):,}")
        val_shards = create_shards(val_tokens, output_dir, "val")
        print(f"Finished writing {val_shards} validation shards.")
    del val_tokens

    meta_path = os.path.join(output_dir, 'meta.pkl')
    print(f"\nSaving metadata to {meta_path}...")

    try:
        enc_meta = tiktoken.get_encoding(tiktoken_model_name)
        padded_vocab_size = n64(enc_meta.n_vocab)

        metadata = {
            'vocab_size': padded_vocab_size,
            'block_size': 1024,
            'tokenizer': tiktoken_model_name,
            'num_train_shards': train_shards if 'train_shards' in locals() else 0,
            'num_val_shards': val_shards if 'val_shards' in locals() else 0,
        }

        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)

        print("Metadata saved successfully:")
        print(metadata)
        print(f"Padded vocab_size: {padded_vocab_size}, Original vocab_size: {original_vocab_size}") # Print both for clarity

    except Exception as e:
        print(f"Error saving metadata: {e}", file=sys.stderr)

    print(f"\nTokenization complete. Output shards in: {output_dir}")
    print(f"Total shards created: {metadata.get('num_train_shards', 0)} (train) + {metadata.get('num_val_shards', 0)} (val)")
    print(f"Remember to update 'vocab_size' in your `config.py` to: {metadata.get('vocab_size')}")
