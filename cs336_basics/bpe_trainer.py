from __future__ import annotations

import os
import time
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor
import multiprocessing as mp
import numpy as np
import regex as re
from queue import PriorityQueue
from collections import defaultdict

from cs336_basics.pretokenization_example import find_chunk_boundaries

pretokens: list[tuple[list[bytes], int]] = []


class ReverseOrder:
    def __init__(self, val: tuple[int, tuple[bytes, bytes]]) -> None:
        self.val = val

    def __lt__(self, other: ReverseOrder) -> bool:
        return self.val > other.val

    def __repr__(self) -> str:
        return repr(self.val)


def pretokenize_chunk(
    pair: tuple[str, list[str]],
) -> dict[bytes, int]:
    """
    Pre-tokenize a chunk of text and update the vocabulary and merges.
    This function should be parallelized across chunks.
    """
    chunk, special_tokens = pair
    chunk = chunk.strip()
    splits = re.split("|".join(special_tokens), chunk)
    tokens: dict[bytes, int] = {}
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for part in splits:
        if not part:
            continue
        # Tokenize the part using the regex pattern
        for token in re.finditer(pat, part):
            token = token.group(0).encode("utf-8")
            tokens[token] = tokens.get(token, 0) + 1
    print(".", end="", flush=True)  # Print a dot for each chunk processed
    return tokens


def pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int = mp.cpu_count(),
    pretokenize_parallel_scale: int = 1,
) -> dict[bytes, int]:
    start_time = time.time()
    inputs = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_processes * pretokenize_parallel_scale,
            "<|endoftext|>".encode("utf-8"),
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            inputs.append((chunk, special_tokens))
    print(f"File read. Pretokenization will be done in {len(inputs)} chunks.")

    with mp.Pool(num_processes) as pool:
        results = pool.map(pretokenize_chunk, inputs)
    tokens = {}
    for ts in results:
        for token, count in ts.items():
            tokens[token] = tokens.get(token, 0) + count
    print(f"tokens of size={len(tokens)}, max count = {max(tokens.values())}")
    pretokenizer_end_time = time.time()
    print(f"Pretokenization took {pretokenizer_end_time - start_time:.2f} seconds")
    return tokens


def get_initial_vocab() -> dict[int, bytes]:
    byte_cardinality = 256
    vocab = {i: bytes([i]) for i in range(byte_cardinality)}
    vocab[byte_cardinality] = "<|endoftext|>".encode("utf-8")
    return vocab


def get_stats(
    pretokens: list[tuple[list[bytes], int]],
    pq: PriorityQueue,
) -> dict[tuple[bytes, bytes], int]:
    pairs = defaultdict(int)
    for symbols, freq in pretokens:
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] += freq
    for pair, freq in pairs.items():
        pq.put(ReverseOrder((freq, pair)))
    return pairs


def get_pretoken_indices_to_examine(
    pair: tuple[bytes, bytes],
    byte_index: dict[bytes, list[int]],
) -> list[int]:
    """
    Get the indices of pretokens that contain every byte element of the pair.
    This is used to reduce the number of pretokens that need to be examined
    when merging pairs.
    """
    candidates = pair[0] + pair[1]
    candidate_list = [bytes([x]) for x in list(candidates)]
    result = byte_index.get(candidate_list[0], set())
    if len(candidate_list) == 1:
        return list(result)
    for candidate in candidate_list[1:]:
        cur_indices = byte_index.get(candidate, set())
        result = result.intersection(cur_indices)
    return list(result)


def merge_pretokens(
    pair: tuple[bytes, bytes],
    pairs: dict[tuple[bytes, bytes], int],
    pq: PriorityQueue,
    byte_index: dict[bytes, list[int]],
) -> None:
    global pretokens
    indices = get_pretoken_indices_to_examine(pair, byte_index)
    # print(f"TEST: heuristic byte {heuristic_byte} for pair {pair}")
    for idx in indices:
        symbols, freq = pretokens[idx]
        found = False
        for i in range(len(symbols) - 1):
            if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                found = True
                break
        if not found:
            continue

        # Update symbols
        new_symbols = []
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(pair[0] + pair[1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        if i == len(symbols) - 1:
            new_symbols.append(symbols[-1])
        pretokens[idx] = (new_symbols, freq)

        # update pair stats
        delta = defaultdict(int)
        for i in range(len(symbols) - 1):
            cur_pair = (symbols[i], symbols[i + 1])
            delta[cur_pair] -= freq
        for i in range(len(new_symbols) - 1):
            cur_pair = (new_symbols[i], new_symbols[i + 1])
            delta[cur_pair] += freq
        for cur_pair, delta_freq in delta.items():
            if delta_freq == 0:
                continue
            pairs[cur_pair] += delta_freq
            pq.put(ReverseOrder((pairs[cur_pair], cur_pair)))


def merge_pretokens_worker(
    input_tuple: tuple[tuple[bytes, bytes], list[int]],
) -> tuple[dict[tuple[bytes, bytes], int], dict[int, tuple[list[bytes], int]]]:
    delta = defaultdict(int)
    sparse_pretokens = {}
    pair, indices = input_tuple
    global pretokens
    assert len(pretokens) > 0, "Pretokens must be initialized before merging."
    for i in range(len(indices)):
        idx = indices[i]
        symbols, freq = pretokens[idx]
        found = False
        for i in range(len(symbols) - 1):
            if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                found = True
                break
        if not found:
            continue

        # Update symbols
        new_symbols = []
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(pair[0] + pair[1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        if i == len(symbols) - 1:
            new_symbols.append(symbols[-1])
        sparse_pretokens[idx] = (new_symbols, freq)

        # update delta stats
        for i in range(len(symbols) - 1):
            cur_pair = (symbols[i], symbols[i + 1])
            delta[cur_pair] -= freq
        for i in range(len(new_symbols) - 1):
            cur_pair = (new_symbols[i], new_symbols[i + 1])
            delta[cur_pair] += freq
    return delta, sparse_pretokens


def merge_pretokens_multiprocess(
    pair: tuple[bytes, bytes],
    pairs: dict[tuple[bytes, bytes], int],
    pq: PriorityQueue,
    byte_index: dict[bytes, list[int]],
    num_processes: int = mp.cpu_count(),
) -> None:
    indices = get_pretoken_indices_to_examine(pair, byte_index)
    global pretokens
    # print(f"TEST: heuristic byte {heuristic_byte} for pair {pair}")
    if not indices:
        return

    chunk_size = len(indices) // num_processes + 1
    index_chunks = [
        indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)
    ]
    inputs = [(pair, idx_chunk) for idx_chunk in index_chunks]

    with mp.Pool(num_processes) as pool:
        results = pool.map(merge_pretokens_worker, inputs)
    for delta, sparse_pretokens in results:
        # Update pairs with delta
        for cur_pair, delta_freq in delta.items():
            if delta_freq == 0:
                continue
            pairs[cur_pair] += delta_freq
            pq.put(ReverseOrder((pairs[cur_pair], cur_pair)))

        # Update pretokens with sparse_pretokens
        for idx, (new_symbols, freq) in sparse_pretokens.items():
            pretokens[idx] = (new_symbols, freq)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    num_processes = kwargs.get("num_processes", mp.cpu_count())
    pretokenize_parallel_scale = kwargs.get("pretokenize_parallel_scale", 1)
    print(f"running BPE with {num_processes} processes")
    tokens = pretokenize(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=num_processes,
        pretokenize_parallel_scale=pretokenize_parallel_scale,
    )

    # Initialize the vocabulary with single-byte tokens and the end-of-text token.
    merges = []
    vocab = get_initial_vocab()

    assert (
        len(vocab) <= vocab_size
    ), f"Vocab size {len(vocab)} is larger than requested {vocab_size}."

    global pretokens
    pretokens = [
        ([bytes([x]) for x in list(token)], count) for token, count in tokens.items()
    ]

    byte_index = defaultdict(set)
    for idx in range(len(pretokens)):
        symbols, _ = pretokens[idx]
        for symbol in symbols:
            byte_index[symbol].add(idx)

    pq = PriorityQueue()
    pairs = get_stats(pretokens, pq)
    while len(vocab) < vocab_size:
        # We need to merge tokens until we reach the desired vocab size
        if not pairs:
            break
        # Filter out invalid stats in PriorityQueue and get the current best pair.
        while not pq.empty():
            freq, best = pq.get().val
            if freq > 0 and pairs[best] == freq:
                break
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]
        merge_pretokens(
            pair=best,
            pairs=pairs,
            pq=pq,
            byte_index=byte_index,
        )
    print(f"vocab size is {len(vocab)}, merges size is {len(merges)}")
    return vocab, merges
