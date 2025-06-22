from __future__ import annotations

import pickle
from collections.abc import Iterable
import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[bytes, int],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_tokens = sorted(self.special_tokens, reverse=True)
        self.vocab_lookup: dict[bytes, int] = {v: k for k, v in vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """
        Load a tokenizer from vocab and merges files.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[str]:
        """
        Encode a string into a list of tokens.
        """
        pretokens: list[bytes] = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # Tokenize the part using the regex pattern
        if not self.special_tokens:
            splits: list[str] = [text]
        else:
            splits: list[str] = re.split("|".join([re.escape(x) for x in self.special_tokens]), text)
        cur_idx: int = 0
        for part in splits:
            for token in re.finditer(PAT, part):
                token: bytes = token.group(0).encode("utf-8")
                pretokens.append(token)
            cur_idx += len(part)
            # print(f"Dealt with part: {part}, current index: {cur_idx}")
            if cur_idx < len(text):
                for special_token in self.special_tokens:
                    window_size = len(special_token)
                    if text[cur_idx : cur_idx + window_size] == special_token:
                        pretokens.append(special_token.encode("utf-8"))
                        cur_idx += window_size
                        break
        # print(
        #     f"Tokenized text of length {len(text)} into {len(pretokens)} pretokens: {pretokens[:20]}..."
        # )
        results: list[int] = []
        for pretoken in pretokens:
            token_id: int | None = self.vocab_lookup.get(pretoken)
            if token_id is not None:
                results.append(token_id)
                continue
            symbols: list[bytes] = [bytes([x]) for x in list(pretoken)]
            for merge in self.merges:
                if len(symbols) <= 2:
                    break
                for i in range(len(symbols) - 2, -1, -1):
                    if symbols[i] == merge[0] and symbols[i + 1] == merge[1]:
                        symbols[i] = merge[0] + merge[1]
                        del symbols[i + 1]
            for symbol in symbols:
                results.append(self.vocab_lookup[symbol])
        # print(f"Encoded {text[:20]}... to {len(results)} tokens.")
        # for idx in results[:5]:
        #     print(self.vocab[idx], end=" ")
        # print()
        return results

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[list[int]]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a string.
        """
        tokens: list[bytes] = [self.vocab.get(i, "\ufffd") for i in ids]
        return b"".join(tokens).decode("utf-8", errors="replace")
