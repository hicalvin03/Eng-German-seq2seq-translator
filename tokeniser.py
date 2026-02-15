from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk
from tokenizers import SentencePieceBPETokenizer
import os


dataset = load_from_disk("cleaned_wmt14")

if not os.path.exists("./trained_tokeniser"):    
    print("training a new tokeniser!")

    untrained_tokeniser = SentencePieceBPETokenizer()

    # use python generator:
    def get_training_corpus():
        train_ds = dataset["train"]
        for idx in range(0,len(train_ds),1000):
            batch = train_ds[idx:idx+1000]["translation"] # grab a batch of {de:" ", en: " "} text

            en_samples = [i["en"] for i in batch] # want to return a list of text ["hello", "friend", ...]
            de_samples = [i["de"] for i in batch]
            total_samples = en_samples + de_samples
            yield total_samples 

    special_tokens = ["<unk>", "<pad>", "<s>", "</s>"] # <s> = start of sentence, </s> = end of sentence

    untrained_tokeniser.train_from_iterator(
        get_training_corpus(), 
        vocab_size = 32000,
        special_tokens=special_tokens
        )
    
    tokeniser = PreTrainedTokenizerFast( # turn a tokeniser.Tokenizer object into pretrainedTokenizer
        tokenizer_object=untrained_tokeniser,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )

    tokeniser.save_pretrained("./trained_tokeniser")

# Tokenise the dataset now

tokeniser = PreTrainedTokenizerFast.from_pretrained("./trained_tokeniser") # wrapping tokeniser in  transformer library. (important for pytorch)

def tokenise_batch(batch):
    batch_text = batch["translation"]
    inputs  = [i["en"] for i in batch_text]
    targets = [i["de"] for i in batch_text]
    return tokeniser(inputs, text_target=targets, truncation=True,max_length= 128) # enforce shorter sentences since bilstm has small context window

dataset = dataset.map(tokenise_batch, batched=True, remove_columns="translation")
dataset.save_to_disk("./tokenised_wmt14")