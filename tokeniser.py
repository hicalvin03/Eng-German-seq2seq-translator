from transformers import AutoTokenizer
from datasets import load_from_disk
import os

old_tokeniser = AutoTokenizer.from_pretrained("gpt2") # uses BPE algorithm borrow GPT2 Template

dataset = load_from_disk("cleaned_wmt14")

# use python generator:
def get_training_corpus():
    train_ds = dataset["train"]
    for idx in range(0,len(train_ds),1000):
        batch = train_ds[idx:idx+1000]["translation"] # grab a batch of {de:" ", en: " "} text

        en_samples = [i["en"] for i in batch] # want to return a list of text ["hello", "friend", ...]
        de_samples = [i["de"] for i in batch]
        total_samples = en_samples + de_samples
        yield total_samples 

# if we haven't trained tokeniser yet train it:
if not os.path.exists("./Trained_Tokeniser"):

    print("Training a new tokeniser")
    new_tokeniser = old_tokeniser.train_new_from_iterator(get_training_corpus(), vocab_size = 32000)
    new_tokeniser.save_pretrained("./Trained_Tokeniser")
    print("finsihed training tokeniser")


# Tokenise data now:
tokeniser = AutoTokenizer.from_pretrained("./Trained_Tokeniser") 

def tokenise_batch(batch):
    batch_text = batch["translation"]

    inputs = [i["en"] for i in batch_text]
    
    targets = [i["de"] for i in batch_text]
    
    return tokeniser(inputs, text_target=targets, truncation=True) # truncation does padding for dataloader to use later


dataset = dataset.map(tokenise_batch,batched=True,remove_columns= "translation") # Map over batches and remove translation column so we get {inputids, attention_mask, label}
dataset.save_to_disk("./tokenised_wmt14")





