# Functions used for training.
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import PAD_IDX,max_length,val_rate,PATH,device

def train(model,optimiser,scheduler,writer,train_dataloader,val_dataloader,state):
    model.zero_grad() 
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    start_epoch = state["epoch"]
    global_step = state["global_step"]
    num_additional_epochs = state["num_additional_epochs"]
    total_train_loss = 0

    for epoch in tqdm(range(start_epoch, start_epoch + num_additional_epochs), desc="epochs"):
        for batch in tqdm(train_dataloader, desc="batch", leave=False):
            loss = batch_process(batch, model, loss_fn)
            
            global_step += 1
            total_train_loss += loss.item()

            loss.backward()
            log_gradients(model,global_step, writer)

            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

            if global_step % val_rate == 0:
                avg_train_loss = total_train_loss / val_rate
                total_train_loss = 0

                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        val_loss = batch_process(batch, model, loss_fn)
                        total_val_loss += val_loss.item()

                writer.add_scalars("Loss", {
                    "train": avg_train_loss,
                    "val": total_val_loss / len(val_dataloader)
                }, global_step)

                lr = optimiser.param_groups[0]['lr']
                writer.add_scalar("LR",lr,global_step)
                model.train()

    writer.flush()
    writer.close()

    
    # Save new checkpoint
    torch.save({
        'epoch': start_epoch + num_additional_epochs,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'writer_path':f"{writer.log_dir}"
    }, f'models/checkpoint_epoch{start_epoch + num_additional_epochs}.pt')

    # Save just model:
    torch.save(model.state_dict(), PATH)

def log_gradients(model,global_step:int,writer,record_rate=19000):
    if global_step % record_rate == 0:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        writer.add_scalar('grad/global_norm', total_norm, global_step)
        writer.add_histogram('grad/enc_l0', model.encoder.bilstm.weight_hh_l0.grad, global_step)
        writer.add_histogram('grad/enc_l1', model.encoder.bilstm.weight_hh_l1.grad, global_step)
        writer.add_histogram('grad/dec_l0', model.decoder.lstm.weight_hh_l0.grad, global_step)
        writer.add_histogram('grad/dec_l1', model.decoder.lstm.weight_hh_l1.grad, global_step)
    
    return

def batch_process(batch,model,loss_fn):

    input_ids = batch["input_ids"].to(device) # (B,L)
    attention_mask = batch["attention_mask"].to(device) # (B,L)
    labels = batch["labels"].to(device) # (B,label_L)

    target_input = insert_pad_token_index(labels)

    Logits = model(input_ids,max_length,attention_mask,target_input)
    B, L, V = Logits.shape

    loss = loss_fn(
        Logits.view(B*L, V),
        labels.view(B*L)
    )
    return loss

def insert_pad_token_index(labels:torch.tensor):
    target_input = labels.clone()
    target_input[target_input == -100] = PAD_IDX
    return target_input

def load_checkpoint_if_exists(model, optimizer, file_path, device):
    start_epoch = 0
    global_step = 0
    writer_path = "runs/new_run"

    if os.path.exists(file_path):
        checkpoint = torch.load(file_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimiser_state_dict'])

        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        writer_path = checkpoint['writer_path']

        print(f"Resuming from epoch {start_epoch}, global_step {global_step}")
    else:
        print("Starting training from scratch.")

    return start_epoch, global_step, writer_path