# Eng-German-seq2seq-translator
Create an Eng-German BILSTM model that uses Luong (multiplicative) attention. Will test with different features such as layernorm, beamsearch and tokenization. 

Model will serve as a nice benchmark to a future transformer translator or fine Tuning a pretrained transformer.

**Model Architecture/details:**

- Dataset: WMT14 from HuggingFace size: ~ 1 million
- Tokeniser sentencepiece BPE vocab_size: 16000
- 2 layer Bilstm + multiplicative attention
- used beam search with width: 10
- hidden_size: 256
- For more details on hyperparams check config.py

**Training Details:**
- Base lr = 3e-4
- Used 1% warmup + cosine annealing scheduling
- adam optimiser
- Used dropout 0.3 and label smoothing 0.1
- Always used teacher forcing

<img width="249" height="251" alt="image" src="https://github.com/user-attachments/assets/a8175139-e26f-4edd-87e2-cd2528b1047b" />

**Testing**
- Training/val loss CrossEntropyLoss
- Testing loss: BLEU 

**Perfomance:**
- Final BLEU: 14.01
- Final crossentropy val loss: 3.6
- Note I suspect low BLEU is due to computational limits as hidden_size and embedding_dim were very small.
<img width="399" height="385" alt="image" src="https://github.com/user-attachments/assets/1cfdc75a-f936-4b0a-bcf5-8959342bc13b" />

**Attention:**
- Observing Attention heatmaps the attention effectively learnt to prioritise the same index token of the encoder for context.
<img width="352" height="346" alt="image" src="https://github.com/user-attachments/assets/b804098e-d12e-4389-a977-aee1abfa4b0f" />


Future improvements:
- Implement scheduled sampling as my model was trained with teacher forcing always.





