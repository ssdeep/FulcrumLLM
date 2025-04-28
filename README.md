# FulcrumLLM
FulcrumLLM is an attempt at creating comprehensive libraries and tooling 
in Scala for LLM/GenAI use cases. This is intended as an educational and training purposes 
to aid in a self-guided programming for learning how to build an LLM from scratch.


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