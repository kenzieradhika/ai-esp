"""Build an Indonesian chat corpus for the PLE trainer.

Source: daruokta/t5gemma2-indonesia-chat-formatted (chat_sft split), public on
HuggingFace: multi-turn conversations in Bahasa Indonesia. Each sample is a
history block followed by a "Jawaban:" line, so at inference the device can
prompt the model with "user: ...\\nJawaban:" and let it continue.

Emits (alongside the English TinyStories bins, suffix "_id"):
  bpe_id32768.json   BPE tokenizer trained on the Indonesian text
  train_id_v32768.bin / val_id_v32768.bin  uint16 token bins
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import requests
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_URL = ("https://huggingface.co/datasets/daruokta/t5gemma2-indonesia-chat-formatted/"
             "resolve/main/chat_sft/train-00000-of-00001.parquet")
VAL_URL = ("https://huggingface.co/datasets/daruokta/t5gemma2-indonesia-chat-formatted/"
           "resolve/main/chat_sft/validation-00000-of-00001.parquet")
VOCAB_SIZE = 32768
RAW_TRAIN = os.path.join(HERE, "id_train.parquet")
RAW_VAL = os.path.join(HERE, "id_val.parquet")
TXT = os.path.join(HERE, "id_chat.txt")
TOK = os.path.join(HERE, f"bpe_id{VOCAB_SIZE}.json")
VAL_FRACTION = 0.005


def download():
    if os.path.exists(RAW_TRAIN) and os.path.exists(RAW_VAL):
        print(f"already have {RAW_TRAIN} and {RAW_VAL}")
        return
    for path, url in ((RAW_TRAIN, TRAIN_URL), (RAW_VAL, VAL_URL)):
        if os.path.exists(path):
            continue
        print(f"downloading {os.path.basename(path)}...", flush=True)
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            n = 0
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
        print(f"  {n / 1e6:.1f}MB")


def format_text():
    if os.path.exists(TXT):
        print(f"already have {TXT}")
        return
    df = pd.concat([pd.read_parquet(RAW_TRAIN), pd.read_parquet(RAW_VAL)], ignore_index=True)
    print(f"rows: {len(df)}")
    docs = []
    for _, row in df.iterrows():
        src = str(row["input"])
        tgt = re.sub(r"<unused\d+>", "", str(row["target"])).strip()
        lines = src.split("\n")
        keep = [ln for ln in lines if not ln.strip().startswith(("system:", "System:"))]
        if not keep or not tgt:
            continue
        docs.append("\n".join(keep).strip() + "\nJawaban: " + tgt + "\n<|endoftext|>")
    with open(TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(docs))
    print(f"docs: {len(docs)}, {os.path.getsize(TXT) / 1e6:.1f}MB")


def train_tokenizer(text):
    if os.path.exists(TOK):
        print(f"already have {TOK}")
        return Tokenizer.from_file(TOK)
    print(f"training BPE vocab={VOCAB_SIZE}...")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator([text[: 40 * 1024 * 1024]], trainer=trainer)
    tok.save(TOK)
    return tok


def main():
    download()
    format_text()
    with open(TXT, "r", encoding="utf-8") as f:
        text = f.read()
    tok = train_tokenizer(text)
    eot = tok.token_to_id("<|endoftext|>")
    print(f"eot id = {eot}")

    print("encoding...")
    docs = text.split("<|endoftext|>")
    ids = []
    for i in range(0, len(docs), 20000):
        batch = [d for d in docs[i : i + 20000] if d.strip()]
        for enc in tok.encode_batch(batch):
            ids.extend(enc.ids)
            ids.append(eot)
        print(f"  {i + len(batch)}/{len(docs)} docs, {len(ids) / 1e6:.1f}M tokens", flush=True)

    dtype = np.uint16 if VOCAB_SIZE <= 65536 else np.uint32
    arr = np.array(ids, dtype=dtype)
    assert arr.max() < VOCAB_SIZE
    n_val = int(len(arr) * VAL_FRACTION)
    arr[:-n_val].tofile(os.path.join(HERE, f"train_v{VOCAB_SIZE}_id.bin"))
    arr[-n_val:].tofile(os.path.join(HERE, f"val_v{VOCAB_SIZE}_id.bin"))
    print(f"train {len(arr) - n_val:,} tokens / val {n_val:,} tokens")
    print(f"compression: {len(text) / len(arr):.2f} bytes/token")


if __name__ == "__main__":
    sys.exit(main())
