# Eng-German-seq2seq-translator
Create an Eng-German BILSTM model that uses Luong (multiplicative) attention. (Learning Project) Will play around with different features such as layernorm, beamsearch and tokenization. 

Model will serve as a nice benchmark to a future transformer translator or fine Tuning a pretrained transformer.

**Model Architecture/details:**

- Dataset: WMT14 from HuggingFace size: ~ 1 million
- Tokeniser sentencepiece BPE vocab_size: 32000
- 2 layer Bilstm.

**Testing**
- Training/val loss CrossEntropyLoss
- Testing loss: BLEU


