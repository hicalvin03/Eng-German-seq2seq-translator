import torch
import torch.nn as nn
from config import SOS_IDX,EOS_IDX

# model definition:
class Bilstm_Encoder(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_size,embedding_matrix):
        super().__init__()
        # input (B,L)
        self.embedding_matrix = embedding_matrix # (B,L,emb)
        self.bilstm = nn.LSTM(input_size=embedding_dim,num_layers=2, hidden_size= hidden_size, bidirectional=True, batch_first=True,dropout=0.3) # takes in (B,L,emb)
        self.bridge_h = nn.Linear(hidden_size*2,hidden_size)
        self.bridge_c = nn.Linear(hidden_size*2,hidden_size)

    def forward(self,input): # (B,L)
        embedded = self.embedding_matrix(input) # Returns (B,L,embedding_dim) 
        output, (h_n,c_n) = self.bilstm(embedded)
        # output: (B,L,2*hidden_size) outputs h_t fwd backward concatenated for the top layer, h_n = (D*num_layers,B,hidden_size) outputs h_n fwd, backward for every layer
        # Layer 1: concatenate fwd and backward h_0,h_n,c_0,c_n
        h_l1 = torch.cat([h_n[0], h_n[1]], dim=1) # (B, enc_hid*2)
        c_l1 = torch.cat([c_n[0], c_n[1]], dim=1) # (B, enc_hid*2)
        # Layer 2:
        h_l2 = torch.cat([h_n[2], h_n[3]], dim=1) # (B, enc_hid*2)
        c_l2 = torch.cat([c_n[2], c_n[3]], dim=1) # (B, enc_hid*2)

        decoder_h_0 = torch.tanh(torch.stack([self.bridge_h(h_l1), self.bridge_h(h_l2)], dim=0)) # (2,B,hidden_size)
        decoder_c_0 = torch.tanh(torch.stack([self.bridge_c(c_l1), self.bridge_c(c_l2)], dim=0)) # (2,B,hidden_size)
        return output, decoder_h_0, decoder_c_0

# compute f(hi,sj) for all hi, then softmax over.
class Luong_attention(nn.Module):
    def __init__(self,encoder_dim,decoder_dim): # Output = C_i = (B,2*hidden_size)  | (B,L, 2*hidden_size)
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(
            decoder_dim, encoder_dim).uniform_(-0.1, 0.1)) # (decoder,encoder)
    
     # query @ W @ values^T 
    def forward(self,query,values,attention_mask): # query:(B,L,decoder),values: (B,L,encoder_dim), attention_mask: (B,L)
        transformed_query = query @ self.W #  (B,L,dec)@(dec,enc) = (B,L,enc)
        attention_weights = transformed_query @ values.transpose(1,2)  # (B,L,enc)@(B,encoder_dim,L) = (B,L,L)

        attention_mask = attention_mask.unsqueeze(1) # (B,1,L)
        attention_weights = attention_weights.masked_fill(attention_mask == 0, float('-inf')) # remove any terms with paddings.

        # d_k = query.shape[-1]
        # attention_weights = attention_weights / (d_k ** 0.5)

        attention_scores = torch.softmax(attention_weights,dim=-1) # (B,L,L)

        context_vector = (attention_scores @ values) # (B,L,encoder_dim)
        return context_vector, attention_scores

class lstm_Decoder(nn.Module):
    def __init__(self,embedding_size,encoder_hidden_size,hidden_size,vocab,embedding_matrix):
        super().__init__()
        self.embedding = embedding_matrix # nn.Embedding(vocab_size,embedding_dim)
        self.attention = Luong_attention(encoder_hidden_size*2,hidden_size)
        self.lstm = nn.LSTM(input_size=embedding_size, num_layers=2,hidden_size= hidden_size, batch_first=True,dropout=0.3)
        self.output = nn.Linear(hidden_size + encoder_hidden_size*2,vocab)
    
    def forward(self,decoder_h_0,decoder_c_0,Encoder_Output,max_length,attention_mask,target_tensor): # training mode (Teacher Forcing)
        device = decoder_h_0.device
        batch_size = Encoder_Output.shape[0]

        input = torch.full((batch_size, 1), SOS_IDX,device=device) # (B,1)
        starting_input = self.embedding(input) # (B,1,embedding_size) a slice of <s>
        
        # Target_tensor (B,max_padding_len)
        max_length = target_tensor.shape[1] # Change max_len to max in label batch
        length_input = self.embedding(target_tensor[:, :-1]) # (B,L-1,embedding_dim) Don't include </s> in input
        full_length_input = torch.cat([starting_input,length_input], dim=1) # (B,L,Embedding_dim) added <s> to start of all sequences
        prediction,output,(h_n,c_n),attention_scores = self.forward_step(full_length_input,decoder_h_0,decoder_c_0,Encoder_Output,attention_mask) # Prediction (B,L,vocab)

        return prediction
   
    def forward_step(self,input,h_n,c_n,Encoder_Output,attention_mask):
        # input: (B,L,input_size)

        query,(h_n,c_n) = self.lstm(input,(h_n,c_n)) # output (B,L,dec_dim)
        context_n, attention_scores = self.attention(query,Encoder_Output,attention_mask) # (B,L,enc*2)

        h_i_context_n = torch.cat([query,context_n],dim=-1) # (B,L,enc*3)
        prediction = self.output(h_i_context_n) # (B,L,vocab)

        return prediction,query,(h_n,c_n),attention_scores
    
    def greedy(self,decoder_h_0,decoder_c_0,Encoder_Output,max_length,attention_mask,target_tensor=None):
        device = decoder_h_0.device
        batch_size = Encoder_Output.shape[0]

        input = torch.full((batch_size, 1), SOS_IDX,device=device) # (B,1)
        starting_input = self.embedding(input) # (B,1,embedding_size) a slice of <s>
        curr_input = starting_input # (B,1,embedding_size)
        logits = []

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device).unsqueeze(1) # (B,1)
        curr_h = decoder_h_0
        curr_c = decoder_c_0

        for i in range(max_length):
            # encoder_output: (B,L,2*encoder_dim)
            prediction, output, (h_n,c_n) = self.forward_step(curr_input,curr_h,curr_c,Encoder_Output,attention_mask)
            logits.append(prediction) # prediction (B,1,vocab)
            curr_h = h_n
            curr_c = c_n

            values,indices = torch.topk(prediction,1,dim=-1) # indices (B,1,1)
            indices = indices.squeeze(-1) # (B,1)
            finished |= (indices == EOS_IDX) # (B,1) keeps track of which sequences are finished

            curr_input = self.embedding(indices) # (B,1,embedding)
            if finished.all():
                break

        return torch.cat(logits,dim=1) # (B,L,vocab)


    def beam_search(self,beam_width,decoder_h_0,decoder_c_0,Encoder_Output,max_length,attention_mask):
        device = decoder_h_0.device

        # Encoder_Output (B,L,2*encoder_dim)
        # Attention_Mask (B,L)
        Encoder_Output = Encoder_Output.repeat(beam_width, 1, 1) # (beam,L,2*encoder_dim)
        attention_mask = attention_mask.repeat(beam_width, 1) # (beam,L)

        input = torch.full((beam_width,1), SOS_IDX,device=device) # (Beam,1)
        starting_input = self.embedding(input) # (beam,1,embedding_size) a slice of <s>
        curr_input = starting_input # (Beam,1,embedding_size)

        curr_beam_scores = torch.full((beam_width,), -1e9, device=device) # (Beam)
        curr_beam_scores[0] = 0.0
        
        # history of current sequences
        history = input # (beam,1)
        attention_history = torch.empty(beam_width, 1, Encoder_Output.shape[1]).to(device) # (beam,1,L)
        attention_history = []

        curr_h = decoder_h_0.repeat(1, beam_width, 1) # (2,beam,hidden_size)
        curr_c = decoder_c_0.repeat(1, beam_width, 1) # (2,beam,hidden_size)

        for i in range(max_length):
            # encoder_output: (beam,L,2*encoder_dim)
            logits, output, (h_n,c_n), attention_scores = self.forward_step(curr_input,curr_h,curr_c,Encoder_Output,attention_mask)
            vocab_size = logits.shape[-1]
            # Logits (beam,1,vocab)
            softmax = nn.LogSoftmax(dim=-1)
            log_probs = softmax(logits).squeeze(1) #(beam,vocab)
            
            # compute conditional prob
            next_scores = log_probs + curr_beam_scores.unsqueeze(-1)  # (beam,vocab)+(beam,1) = (beam,vocab)
            next_scores = next_scores.flatten() # (beam*vocab)

            values,indices = torch.topk(next_scores,beam_width,dim=-1) # values,indices (beam_width)

            beam_indices = torch.div(indices, vocab_size, rounding_mode='floor')  # (beam_width) - which previous beam
            token_indices = indices % vocab_size  # (beam_width) - which token was chosen
            
            # Update histories:
            curr_beam_scores = values # (beams)
            history = torch.cat([history[beam_indices], token_indices.unsqueeze(1)], dim=1) # (beam,1), (beam,1)
            curr_input = self.embedding(token_indices.unsqueeze(1)) # (beam,1,embedding)

            # attention_history = torch.cat([attention_history,attention_scores[beam_indices,:,:]],dim=1) #(beam,iteration,L)
            attention_history.append(attention_scores[beam_indices,:,:])
            
            curr_h = h_n[:, beam_indices, :] # (2,beam,hidden_size)
            curr_c = c_n[:, beam_indices, :] # (2,beam,hidden_size)

        
        best_sequence_index = torch.argmax(curr_beam_scores)
        best_sequence = history[best_sequence_index]
        # best_sequence = best_sequence.view(1,-1)
        attention_history = torch.cat(attention_history,dim=1)
        best_attention_score = attention_history[best_sequence_index] # (1,L,L)

        return best_sequence,best_attention_score # (1,L)

            
class seq2seq_bilstm(nn.Module):
    def __init__(self,vocab_size,embedding_dim,encoder_hidden,decoder_hidden): # Output = C_i = (B,2*hidden_size) 
        super().__init__()

        self.embedding_matrix = nn.Embedding(vocab_size,embedding_dim) # (B,L,emb) shared embedding matrix
        self.encoder = Bilstm_Encoder(vocab_size,embedding_dim,encoder_hidden,self.embedding_matrix)
        self.decoder = lstm_Decoder(embedding_dim,encoder_hidden,decoder_hidden,vocab_size,self.embedding_matrix)

    def forward(self,input,max_length,attention_mask,target_tensor):
        output, decoder_h_0, decoder_c_0 = self.encoder(input)
        Logits = self.decoder(decoder_h_0,decoder_c_0,output,max_length,attention_mask,target_tensor) # (B,L,Vocab)
        
        return Logits # (B,L,Vocab)
    
    @torch.no_grad()
    def generate(self,input,max_length,attention_mask,beam_width): # returns list shape (b,L)
        output, decoder_h_0, decoder_c_0 = self.encoder(input)
        
        if (beam_width == 1): # greedy
            Logits = self.decoder.greedy(decoder_h_0,decoder_c_0,output,max_length,attention_mask) # (b,L,vocab)
            predictions = torch.argmax(Logits,dim=-1).tolist() # list of sentences (B,L)
            return predictions
        else:
            predictions,attention_score = self.decoder.beam_search(beam_width,decoder_h_0,decoder_c_0,output,max_length,attention_mask)
            return predictions.tolist(),attention_score # (b,L) predictions







