import time
import tiktoken
from tokenizers import Tokenizer
from tqdm import tqdm

def benchmark_tokenizer(name, encode_fn, decode_fn, texts):
    total_tokens = 0
    incorrect_tokenization = 0
    start_time = time.time()

    for text in tqdm(texts, desc=f"Benchmarking {name}"):
        tokens = encode_fn(text)
        decoded = decode_fn(tokens)

        if text != decoded:
            incorrect_tokenization += 1

        total_tokens += len(tokens)

    duration = time.time() - start_time
    throughput = total_tokens / duration

    print(f"--- {name} ---")
    print(f"Total tokens: {total_tokens}")
    print(f"Time taken: {duration:.3f} seconds")
    print(f"Throughput: {throughput:.2f} tokens/sec")
    print(f"Incorrect tokenizations: {incorrect_tokenization}\n")

def main():
    # Load sample texts
    with open("/Users/dooby1/Downloads/wikiarchive/AllCombined.txt", "r", encoding="utf-8") as f:
        texts = f.read().splitlines()

    # OpenAI o200k_base Tokenizer
    tiktoken_enc = tiktoken.get_encoding("o200k_base")

    def tiktoken_encode(text):
        return tiktoken_enc.encode(text)

    def tiktoken_decode(tokens):
        return tiktoken_enc.decode(tokens)

    # Huggingface Tokenizer (GPT-2, as a baseline)
    hf_tokenizer = Tokenizer.from_pretrained("gpt2")

    def hf_encode(text):
        return hf_tokenizer.encode(text).ids

    def hf_decode(tokens):
        return hf_tokenizer.decode(tokens)

    # Benchmark
    benchmark_tokenizer("Tiktoken (o200k_base)", tiktoken_encode, tiktoken_decode, texts)
    benchmark_tokenizer("Huggingface (gpt2)", hf_encode, hf_decode, texts)

if __name__ == "__main__":
    main()
