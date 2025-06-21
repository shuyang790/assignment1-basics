import json
import time
import cProfile
import io
import pickle

from .adapters import run_train_bpe


def test_large_train_bpe_tinystories():
    """
    Ensure that BPE training works on a larger dataset.
    """
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    start_time = time.time()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    print(f"BPE training took {end_time - start_time:.2f} seconds")
    assert end_time - start_time < 3 * 60  # Ensure it takes less than 3 minutes
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token in vocab: {longest_token} ({len(longest_token)} bytes)")
    with open("data/tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("data/tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)


def test_large_train_bpe_owt():
    """
    Ensure that BPE training works on a larger dataset.
    """
    input_path = "data/owt_train.txt"
    start_time = time.time()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        num_processes=4,
        pretokenize_parallel_scale=20,
    )
    end_time = time.time()
    print(f"BPE training took {end_time - start_time:.2f} seconds")
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token in vocab: {longest_token} ({len(longest_token)} bytes)")
    with open("data/owt_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("data/owt_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
