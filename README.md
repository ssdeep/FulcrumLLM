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

```bash
Loading tokenizer file 100% │██████████████│ 199998/199998 (0:00:01 / 0:00:00) 
Processing text lines 100% │█████████████│ 2052699/2052699 (0:02:32 / 0:00:00) 
Total tokens: 40929433
Time taken: 152.867 seconds
Throughput: 267745.21 tokens/sec
Incorrect tokens counts: 0
```