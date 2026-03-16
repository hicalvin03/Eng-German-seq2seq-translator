# model_path
PATH = "models/bilstm"
CHECKPOINT_PATH = "models/checkpoint_epoch30.pt"
# special Tokens global variables
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
# Model Parameters:
max_length = 40
hidden_size = 256
vocab_size = 16000
embedding_dim = 256
batch_size = 32
beam_width = 10
test_batch_size = 1
lr = 3e-4
num_layers = 2
val_rate = 1000
num_epochs = 20
epochs_per_run = 20 # per training run how many epochs in one go

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")