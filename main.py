# main file used to execute training of model.
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR,LinearLR,CosineAnnealingLR
from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from train import load_checkpoint_if_exists,train
from model import seq2seq_bilstm
from config import batch_size,num_epochs,hidden_size,embedding_dim,vocab_size,device,lr,epochs_per_run,CHECKPOINT_PATH

# section 1 load data:
dataset = load_from_disk("tokenised_wmt14")
dataset = dataset.with_format("torch") # turns the lists to torch tensors

train_dataset = dataset["train"]
test_dataset = dataset["test"]
val_dataset = dataset["validation"]

half = len(train_dataset) // 2 # Reduce Training dataset by half
train_dataset = train_dataset.shuffle(seed=42).select(range(half)) # grab random half to remove bias


tokeniser = PreTrainedTokenizerFast.from_pretrained("./trained_tokeniser") # import my trained tokeniser
collate_fn = DataCollatorForSeq2Seq(tokeniser,padding=True)
#prepare the dataloaders:
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=1,collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size,collate_fn=collate_fn)


# section 2 train the model and save it
total_steps = len(train_dataset)//batch_size * num_epochs
warmup_steps = total_steps*0.01 # warmup for 1%

model = seq2seq_bilstm(vocab_size, embedding_dim, hidden_size, hidden_size).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

warmup = LinearLR(optimiser, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
cosine = CosineAnnealingLR(optimiser, T_max=(total_steps - warmup_steps), eta_min=5e-5)

scheduler = SequentialLR(optimiser, schedulers=[warmup, cosine], milestones=[warmup_steps])

start_epoch, global_step, writer_path = load_checkpoint_if_exists( # start training from a CHECKPOINT_PATH if exists else train from scratch
    model,
    optimiser,
    CHECKPOINT_PATH,
    device
)

writer = SummaryWriter(writer_path) # used for logging training loss

state = { # start state
    "epoch": start_epoch,
    "global_step": global_step,
    "num_additional_epochs": epochs_per_run
}

train(model,optimiser,scheduler,writer,train_dataloader,val_dataloader,state)