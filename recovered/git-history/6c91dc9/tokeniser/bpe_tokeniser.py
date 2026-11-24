from collections import Counter
import json


class BPETokenizer:

    ############################################################
    # Constructor
    ############################################################
    def __init__(self):

        self.special_tokens = [
            "<PAD>",
            "<BOS>",
            "<EOS>",
            "<UNK>"
        ]

        self.vocab = set()
        self.merges = []

        self.token_to_id = {}
        self.id_to_token = {}

    ############################################################
    # Count adjacent pairs
    ############################################################
    def count_pairs(self, corpus):

        pair_counts = Counter()

        for word, freq in corpus.items():

            for i in range(len(word) - 1):

                pair = (word[i], word[i + 1])

                pair_counts[pair] += freq

        return pair_counts

    ############################################################
    # Merge one pair everywhere
    ############################################################
    def merge_pair(self, corpus, pair):

        new_corpus = Counter()

        for word, freq in corpus.items():

            new_word = []

            i = 0

            while i < len(word):

                if (
                    i < len(word) - 1
                    and word[i] == pair[0]
                    and word[i + 1] == pair[1]
                ):

                    new_word.append(pair[0] + pair[1])
                    i += 2

                else:

                    new_word.append(word[i])
                    i += 1

            new_corpus[tuple(new_word)] += freq

        return new_corpus

    ############################################################
    # Build token ids
    ############################################################
    def build_vocab_ids(self):

        self.token_to_id = {}
        self.id_to_token = {}

        # Special tokens first
        for idx, token in enumerate(self.special_tokens):

            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        next_id = len(self.special_tokens)

        # Remaining vocabulary
        for token in sorted(self.vocab):

            if token not in self.token_to_id:

                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token

                next_id += 1
        self.pad_token_id = self.token_to_id["<PAD>"]
        self.bos_token_id = self.token_to_id["<BOS>"]
        self.eos_token_id = self.token_to_id["<EOS>"]
        self.unk_token_id = self.token_to_id["<UNK>"]
    ############################################################
    # Train BPE
    ############################################################
    def train(self, text, target_vocab_size):

        corpus = Counter(
            tuple(list(word) + ["</w>"])
            for word in text.split()
        )

        self.vocab = set()

        for word in corpus:
            self.vocab.update(word)

        self.merges = []

        while len(self.vocab) < target_vocab_size - len(self.special_tokens):

            pair_counts = self.count_pairs(corpus)

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)

            corpus = self.merge_pair(corpus, best_pair)

            new_token = best_pair[0] + best_pair[1]

            self.vocab.add(new_token)

            self.merges.append(best_pair)

        self.build_vocab_ids()

    ############################################################
    # Encode text -> ids
    ############################################################
    def encode(self, text):

        tokens = []

        for word in text.split():

            word_tokens = list(word) + ["</w>"]

            for pair in self.merges:

                merged = []

                i = 0

                while i < len(word_tokens):

                    if (
                        i < len(word_tokens) - 1
                        and word_tokens[i] == pair[0]
                        and word_tokens[i + 1] == pair[1]
                    ):

                        merged.append(pair[0] + pair[1])

                        i += 2

                    else:

                        merged.append(word_tokens[i])

                        i += 1

                word_tokens = merged

            tokens.extend(word_tokens)

        ids = []

        for token in tokens:

            ids.append(
                self.token_to_id.get(
                    token,
                    self.token_to_id["<UNK>"]
                )
            )

        return ids

    ############################################################
    # Decode ids -> text
    ############################################################
    def decode(self, ids):

        text = ""

        for idx in ids:

            token = self.id_to_token[idx]

            if token in (
                "<PAD>",
                "<BOS>",
                "<EOS>"
            ):
                continue

            if token == "<UNK>":
                text += "?"
                continue

            text += token

        text = text.replace("</w>", " ")

        return text.strip()

    ############################################################
    # Save tokenizer
    ############################################################
    def save(self, vocab_path, merges_path):

        with open(vocab_path, "w", encoding="utf-8") as f:

            json.dump(
                self.token_to_id,
                f,
                indent=4
            )

        with open(merges_path, "w", encoding="utf-8") as f:

            json.dump(
                self.merges,
                f,
                indent=4
            )

    ############################################################
    # Load tokenizer
    ############################################################
    def load(self, vocab_path, merges_path):

        with open(vocab_path, "r", encoding="utf-8") as f:

            self.token_to_id = json.load(f)
            self.pad_token_id = self.token_to_id["<PAD>"]
            self.bos_token_id = self.token_to_id["<BOS>"]
            self.eos_token_id = self.token_to_id["<EOS>"]
            self.unk_token_id = self.token_to_id["<UNK>"]

        self.id_to_token = {

            idx: token

            for token, idx in self.token_to_id.items()

        }

        self.vocab = set(self.token_to_id.keys()) - set(
            self.special_tokens
        )

        with open(merges_path, "r", encoding="utf-8") as f:

            self.merges = [
                tuple(pair)
                for pair in json.load(f)
            ]