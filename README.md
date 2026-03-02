# Eng-German-seq2seq-translator
Create an Eng-German BILSTM model that uses Luong (multiplicative) attention. (Learning Project) Will play around with different features such as layernorm, beamsearch and tokenization. 

Model will serve as a nice benchmark to a future transformer translator or fine Tuning a pretrained transformer.

**Model Architecture/details:**

- Dataset: WMT14 from HuggingFace size: ~ 1 million
- Tokeniser sentencepiece BPE vocab_size: 16000
- 2 layer Bilstm + multiplicative attention
- used beam search with width: 10
- hidden_size: 256
- For more details on hyperparams check config.py

** Training Details:**
- Used

**Testing**
- Training/val loss CrossEntropyLoss
- Testing loss: BLEU 

**Perfomance:**
- Final BLEU: 14.01
- Final crossentropy val loss: 3.6




