from __future__ import annotations

import json
import os
import resource
import regex as re
import sys
import time

import psutil
import pytest
import tiktoken

from .adapters import get_tokenizer
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode
from cs336_basics.tokenizer import Tokenizer

VOCAB_PATH = FIXTURES_PATH / "../../data/tinystories_vocab.pkl"
MERGES_PATH = FIXTURES_PATH / "../../data/tinystories_merges.pkl"


def test_tokenizer_experiment_tinystories():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=VOCAB_PATH,
        merges_filepath=MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )
    with open(FIXTURES_PATH / "../../data/TinyStoriesV2-GPT4-valid.txt") as f:
        content = " ".join(f.readlines())
    documents = re.split(re.escape("<|endoftext|>"), content)
    selected = "<|endoftext|>".join(documents[:100])
    start_time = time.time()
    tokens = tokenizer.encode(selected)
    end_time = time.time()
    duration_seconds = end_time - start_time
    print(f"Custom tokenizer took {duration_seconds:.2f} seconds")
    num_bytes = len(list(selected.encode("utf-8")))
    num_tokens = len(tokens)
    print(
        f"Compression rate: {num_bytes / num_tokens:.2f} bytes/token"
    )  # 4.14 bytes / token
    print(
        f"Throughput: {num_bytes / duration_seconds * 1e-6:.2f} MB/second"
    )  # 1.65 MB / second
