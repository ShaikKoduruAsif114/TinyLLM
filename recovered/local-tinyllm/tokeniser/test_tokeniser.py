from bpe_tokeniser import BPETokenizer
import time

tokenizer = BPETokenizer()
tokenizer.load("vocab.json", "merges.json")

# Correctness check
sample = "Once upon a time there was a little girl"
encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

print("Encoded:", encoded)
print("Decoded:", decoded)
print("Match:", sample == decoded)

# Speed check — encode 1000 copies of the sample
big_text = " ".join([sample] * 1000)

start = time.time()
tokenizer.encode(big_text)
elapsed = time.time() - start

print(f"\nEncoded 1000 copies in {elapsed:.2f} seconds")