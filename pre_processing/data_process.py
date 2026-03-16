import pandas as pd
from datasets import load_dataset
import os
import re
os.environ["HF_DATASETS_OFFLINE"] = "1" # don't call the api just use local

dataset = load_dataset("wmt14", "de-en")

def clean(example):
    de_text = example['translation']['de']
    en_text = example['translation']['en']

    # Check 1: check for null/empty
    if not de_text and not en_text:
        return False

    # Check 2: remove long sentences
    if len(de_text) > 150 or len(en_text) > 150:
        return False
    
    return True

def normalise(example):
    for lang in ['de','en']:
        text = example['translation'][lang]
        text = re.sub(r'<.*?>', '', text) # Removes HTML elements
        text = re.sub(r'\s+', ' ', text).strip() # removes excess spaces
        example['translation'][lang] = text

    return example

filtered_dataset = dataset.filter(clean)

normalised_dataset = filtered_dataset.map(normalise)


file_name = "cleaned_wmt14"

normalised_dataset.save_to_disk(file_name)
normalised_dataset.cleanup_cache_files()
print(f"Finished cleaning and saved to: {file_name}")
