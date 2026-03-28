"""
Process ShareGPT dataset to extract input/output token lengths.

Output: datasets/sharegpt_lengths.npy — shape (N, 2), columns = [input_len, output_len]
        Token lengths measured using tiktoken cl100k_base encoding.

Each "request" = first valid (user, assistant) turn pair in a conversation.
User roles: human, user
Assistant roles: gpt, chatgpt, bard, bing
Conversations starting with an assistant turn: skip the leading assistant turn,
then look for the first user-assistant pair.
"""

import json
import numpy as np
import tiktoken
import os
import time

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "ShareGPT_V3_unfiltered_cleaned_split.json"
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sharegpt_lengths.npy")

USER_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"gpt", "chatgpt", "bard", "bing"}


def find_first_pair(turns):
    """Find first (user, assistant) adjacent pair, skipping system/leading assistant turns."""
    for i in range(len(turns) - 1):
        if turns[i]["from"] in USER_ROLES and turns[i + 1]["from"] in ASSISTANT_ROLES:
            return turns[i]["value"], turns[i + 1]["value"]
    return None, None


def main():
    enc = tiktoken.get_encoding("cl100k_base")

    print(f"Loading {DATASET_PATH} ...")
    t0 = time.time()
    with open(DATASET_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversations in {time.time() - t0:.1f}s")

    lengths = []
    skipped = 0
    for conv in data:
        input_text, output_text = find_first_pair(conv["conversations"])

        if not input_text or not output_text:
            skipped += 1
            continue

        input_len = len(enc.encode(input_text, disallowed_special=()))
        output_len = len(enc.encode(output_text, disallowed_special=()))
        lengths.append((input_len, output_len))

    arr = np.array(lengths, dtype=np.int32)
    np.save(OUTPUT_PATH, arr)

    print(f"Saved {len(lengths)} request lengths to {OUTPUT_PATH}")
    print(f"Skipped {skipped} conversations")
    print(
        f"Input  length — min: {arr[:, 0].min()}, max: {arr[:, 0].max()}, "
        f"mean: {arr[:, 0].mean():.1f}, median: {np.median(arr[:, 0]):.1f}"
    )
    print(
        f"Output length — min: {arr[:, 1].min()}, max: {arr[:, 1].max()}, "
        f"mean: {arr[:, 1].mean():.1f}, median: {np.median(arr[:, 1]):.1f}"
    )


if __name__ == "__main__":
    main()
