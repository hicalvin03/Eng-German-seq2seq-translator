# file to execute testing and evaluating.
import torch
from tqdm import tqdm
from torchmetrics.text import SacreBLEUScore
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForSeq2Seq
from train import insert_pad_token_index
from model import seq2seq_bilstm
from datasets import load_from_disk
from config import EOS_IDX,hidden_size,embedding_dim,vocab_size,device,PATH,max_length


# cleanup sentences
def truncate_at_eos(sentence):
    if EOS_IDX in sentence:
        sentence = sentence[:sentence.index(EOS_IDX)]
    return sentence

# Testing: Run BLEU on the model
model = seq2seq_bilstm(vocab_size, embedding_dim, hidden_size, hidden_size).to(device)
model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()

dataset = load_from_disk("tokenised_wmt14")
dataset = dataset.with_format("torch") # turns the lists to torch tensors

test_dataset = dataset["test"]

tokeniser = PreTrainedTokenizerFast.from_pretrained("./trained_tokeniser") # import my trained tokeniser
collate_fn = DataCollatorForSeq2Seq(tokeniser,padding=True)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=1,collate_fn=collate_fn)

model_sentences = []
reference_sentences = []

for batch in tqdm(test_dataloader,desc="batch"):
    input_ids = batch["input_ids"].to(device) # (B,L)
    attention_mask = batch["attention_mask"].to(device) # (B,L)
    labels = batch["labels"].to(device) # (B,label_L)

    # swap the -100's with pad
    target_input = insert_pad_token_index(labels)

    label_sentences = target_input.tolist() # list of correct sentences

    prediction,attention_scores = model.generate(input_ids,max_length,attention_mask,10) # (B,L)
    cleaned_sentences = []
    # for sentence in predictions:
    cleaned_sentences.append(truncate_at_eos(prediction))

    decoded_model_sentences = tokeniser.batch_decode(cleaned_sentences,skip_special_tokens=True)
    decoded_label_sentences = tokeniser.batch_decode(label_sentences,skip_special_tokens=True)

    model_sentences.extend(decoded_model_sentences)
    reference_sentences.extend(decoded_label_sentences)


reference_sentences = [[s] for s in reference_sentences]

bleu = SacreBLEUScore()
bleu_score = bleu(preds=model_sentences,target=reference_sentences)
print(bleu_score)


    