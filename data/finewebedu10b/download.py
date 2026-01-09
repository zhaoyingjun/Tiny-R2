from datasets import load_dataset
import os
import time
from tqdm import tqdm

# Configuration
dataset_name = "HuggingFaceFW/fineweb-edu"
dataset_subset = "sample-10BT"
output_folder = "./fineweb_chunks"  # Folder to save chunk files
chunk_size = 10000  # Number of examples per chunk file

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the dataset (sample-10BT subset)
dataset = load_dataset(dataset_name, dataset_subset, split="train", streaming=True)

file_counter = 0
example_counter = 0
current_file = None

start_time = time.time()  # Start time for overall progress

total_examples = 10_000_000 # Approximate total examples in sample-10BT (adjust if needed for other subsets)

print(f"Downloading and chunking dataset '{dataset_name}/{dataset_subset}'...")

# Wrap dataset iteration with tqdm for progress bar
with tqdm(total=total_examples, desc="Chunking Dataset") as progress_bar:
    for example in dataset:
        text_content = example["text"]

        if example_counter % chunk_size == 0:
            # Close the previous file if it's open
            if current_file:
                current_file.close()

            # Create a new chunk file
            chunk_filename = os.path.join(output_folder, f"chunk_{file_counter:05d}.txt") # 5 digits for chunk number
            current_file = open(chunk_filename, "w", encoding="utf-8")
            progress_bar.write(f"Creating chunk file: {chunk_filename}") # Use progress_bar.write to avoid messing up progress bar
            file_counter += 1

        if text_content: # Check if text content is not None or empty
            current_file.write(text_content)
            current_file.write("\n\n<|file_separator|>\n\n") # Add a separator between examples

        example_counter += 1
        progress_bar.update(1) # Increment progress bar by 1 example

# Close the last file
if current_file:
    current_file.close()

end_time = time.time() # End time
elapsed_time = end_time - start_time # Calculate elapsed time

print(f"Dataset chunking complete.")
print(f"Saved chunk files to '{output_folder}'.")
print(f"Total examples processed: {example_counter}")
print(f"Total chunk files created: {file_counter}")
print(f"Elapsed time: {elapsed_time:.2f} seconds") # Print elapsed time
