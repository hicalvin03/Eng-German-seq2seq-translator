from transformers import AutoTokenizer
from datasets import load_from_disk

old_tokenizer = AutoTokenizer.from_pretrained("gpt2") # uses BPE algorithm borrow GPT2 Template

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


tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), vocab_size = 32000)

tokenizer.save_pretrained("./Trained_Tokeniser")