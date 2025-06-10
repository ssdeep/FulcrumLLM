# FulcrumLLM
FulcrumLLM is an attempt at creating comprehensive libraries and tooling 
in Scala for LLM/GenAI use cases. This is intended as an educational and training purposes 
to aid in a self-guided programming for learning how to build an LLM from scratch.

The goal of this project is to have a self contained repository of all the required tooling and libraries that are typically 
not available in scala for ML systems, more specifically all that is required to build an LLM from scratch

## Reference 
1. This repository is closely following Sebastian Raschka's text book "Building A Large Language Model From Scratch"
More information about this book can be found here
https://www.manning.com/books/build-a-large-language-model-from-scratch

2. Rashka's book is intended for python programmer and much of the content relies on familiarity of or introduction to the use of 
PyTorch libraries. We substitute these with a Scala implementation of PyTorch called Storch. More information about Storch and 
   it's limitations can be found here https://storch.dev/

I will list out more references as I make progress with this course and where necessary will be building substitution tools 
   and libraries as part of the FulcrumLLM project.


## Tokenizer 
The Tokenizer built in this project is a Byte-Pair Encoding tokenizer that uses a simple
prefix trie based approach to tokenize a string. We use the OpenAI tiktoken library and huggingface 
BPE tokenizer as reference.

More info on BPE can be found here https://huggingface.co/learn/llm-course/en/chapter6/5

Our benchmark for tokenizing 
- Tokenization Dataset: https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish
- Pre-loaded tokens from OpenAI's tiktoken 200k vocab list: https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken

Basic Benchmarking that includes
- Encoding: Converting String to Tokens 
- Decoding: Converting Tokens back to String
- Validation: Compare original string and the decoded string and compare all cases where there are mismatches of token strings.
  The aim is to get to 0 mismatches

### Results
#### Benchmarks for the Scala based Fulcrum Tokenizer
```bash
Loading tokenizer file 100% │██████████████│ 199998/199998 (0:00:01 / 0:00:00) 
Processing text lines 100% │█████████████│ 2052699/2052699 (0:02:32 / 0:00:00) 
Total tokens: 40929433
Time taken: 152.867 seconds
Throughput: 267745.21 tokens/sec
Incorrect tokens counts: 0
```

#### Benchmarks for Python based OpenAI Tiktoken and Huggingface gpt2 tokenizer 
```bash
tokenizer.json: 100%|██████████████████████| 1.36M/1.36M [00:00<00:00, 8.91MB/s]
Benchmarking Tiktoken (o200k_base): 100%|█| 2052699/2052699 [01:00<00:00, 33816.
--- Tiktoken (o200k_base) ---
Total tokens: 41073977
Time taken: 60.702 seconds
Throughput: 676653.81 tokens/sec
Incorrect tokenizations: 0

Benchmarking Huggingface (gpt2): 100%|█| 2052699/2052699 [02:59<00:00, 11412.02i
--- Huggingface (gpt2) ---
Total tokens: 40673536
Time taken: 179.872 seconds
Throughput: 226124.85 tokens/sec
Incorrect tokenizations: 0
```

## Multi-head Attention
The Multi-head Attention module is a Scala implementation of the multi-head attention mechanism, which is a key component of the Transformer architecture. It allows the model to focus on different parts of the input sequence simultaneously, improving its ability to capture complex dependencies.
The module includes a step-by-step progression of attention mechanisms following the chapter 4 of Raschka's book, starting from a simple dot-product attention to a more complex multi-head attention mechanism.


## Transformer 
The Transformer module is a Scala implementation of the Transformer architecture, which is the backbone of many modern LLMs. It includes components such as multi-head self-attention, position-wise feed-forward networks, and layer normalization.


## Training
The training code is based on the Storch library and follows the PyTorch API closely. The training loop is designed to be flexible and allows for easy modification of the model, loss function, and optimizer.
Here is an example configuration for training a simple model:

```hocon

```