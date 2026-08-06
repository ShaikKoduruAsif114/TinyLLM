from bpe_tokeniser import BPETokenizer

tokenizer = BPETokenizer()

tokenizer.load("vocab.json", "merges.json")

sample = "Once upon a time"

encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

print("Original :", sample)
print("Encoded  :", encoded)
print("Decoded  :", decoded)
print("Match    :", sample == decoded)