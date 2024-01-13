import pandas as pd

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader


def get_tokenizer(model_checkpoint):
    """
    Get tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return tokenizer


def build(q_len, ctx_len, BE_val_batch_size, no_hard):
    dval = pd.read_csv("valid.csv")
    tokenizer = get_tokenizer(BE_checkpoint)
    print("\t* Loading data...")
    val_loader = build_dpr_traindata(
        df=dval,
        tokenizer=tokenizer,
        q_len=q_len,
        ctx_len=ctx_len,
        batch_size=BE_val_batch_size,
        no_hard=no_hard,
        shuffle=True,
    )


def build_dpr_traindata(
    df, tokenizer, q_len, ctx_len, batch_size, no_hard, shuffle=False, sub=False
):
    """
    This funtion builds train and val data loader for biencoder training
    """
    questions = df["question"].tolist()
    positives = df["answer"].tolist()
    print("question: ", questions[:10])
    print("positives: ", positives[:10])
    Q = tokenizer.batch_encode_plus(
        questions,
        padding="max_length",
        truncation=True,
        max_length=q_len,
        return_tensors="pt",
    )
    P = tokenizer.batch_encode_plus(
        positives,
        padding="max_length",
        truncation=True,
        max_length=ctx_len,
        return_tensors="pt",
    )
    data_tensor = TensorDataset(
        Q["input_ids"], Q["attention_mask"], P["input_ids"], P["attention_mask"]
    )
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=shuffle)
    print(data_loader)
    return data_loader


q_len = 32
ctx_len = 256
BE_val_batch_size = 64
no_hard = 0
BE_checkpoint = "vinai/phobert-base-v2"
build(q_len, ctx_len, BE_val_batch_size, no_hard)
