from datasets import load_dataset
from transformers import AutoTokenizer
import torch
def get_squad_loader(batch_size=4, max_len=384):
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    ds = load_dataset("squad", split="train[:2000]")

    def _tok(ex):
        txt = ex["question"] + " " + ex["context"]
        tokens = tok(txt, truncation=True, padding="max_length", max_length=max_len)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    ds = ds.map(_tok, remove_columns=ds.column_names)
    ds.set_format(type="torch")
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
