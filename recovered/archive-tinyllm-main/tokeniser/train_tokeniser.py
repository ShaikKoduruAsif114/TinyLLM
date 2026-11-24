from datasets import load_dataset
from bpe_tokeniser import BPETokenizer

dataset = load_dataset("roneneldan/TinyStories", split="train")

raw_text = " ".join(dataset[:50000]["text"])

tokenizer = BPETokenizer()

tokenizer.train(raw_text, 4096)

tokenizer.save("vocab.json", "merges.json")

print("Training Complete!")

sample = "once upon a time there was a little girl"
encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

print("Encoded:", encoded)
print("Decoded:", decoded)
print("Match:", sample == decoded)