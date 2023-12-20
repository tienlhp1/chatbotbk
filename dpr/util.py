import random
import json
import pandas as pd
import torch
from torch import Tensor as T
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

def get_tokenizer(model_checkpoint):
    """
    Get tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return tokenizer

def build_dpr_traindata(corpus, df, tokenizer, q_len, ctx_len, batch_size, no_hard, shuffle = False, sub=False):
    """
    This funtion builds train and val data loader for biencoder training
    """
    questions = df["tokenized_question"].tolist()
    if not sub:
        ans_ids = df["ans_id"].tolist()
        neg_ids = df["neg_ids"].tolist()
    else:
        ans_ids = df["ans_sub_id"].tolist()
        neg_ids = df["neg_sub_ids"].tolist() 
    positives = [corpus[i] for i in ans_ids]
    if no_hard != 0:
        negatives = []
        for x in neg_ids:
            str_ids = x[1:-1].split(", ")
            ids = [int(i) for i in str_ids]
            ids = ids[:no_hard]
            neg = [corpus[i] for i in ids]
            negatives += neg
        
    Q = tokenizer.batch_encode_plus(questions, padding='max_length', truncation=True, max_length=q_len, return_tensors='pt')
    P = tokenizer.batch_encode_plus(positives, padding='max_length', truncation=True, max_length=ctx_len, return_tensors='pt')
    if no_hard != 0:
        N = tokenizer.batch_encode_plus(negatives, padding='max_length', truncation=True, max_length=ctx_len, return_tensors='pt')
        N_ids = N['input_ids'].view(-1,no_hard,ctx_len)
        N_attn = N['attention_mask'].view(-1,no_hard,ctx_len)
        data_tensor = TensorDataset(Q['input_ids'], Q['attention_mask'], P['input_ids'], P['attention_mask'], N_ids, N_attn)
    else:
        data_tensor = TensorDataset(Q['input_ids'], Q['attention_mask'], P['input_ids'], P['attention_mask'])
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def build_dpr_testdata(df, tokenizer, q_len, batch_size, shuffle = False, sub=False):
    """
    This funtion builds val and test dataloader for quick evaluating biencoder
    """
    questions = df["tokenized_question"].tolist()
    if not sub:
        ans_ids = df["ans_id"].tolist()
    else:
        ans_ids = df["ans_sub_id"].tolist()
    labels = torch.tensor(ans_ids, dtype=torch.long)
        
    Q = tokenizer.batch_encode_plus(questions, padding='max_length', truncation=True, max_length=q_len, return_tensors='pt')

    data_tensor = TensorDataset(Q['input_ids'], Q['attention_mask'], labels)
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def embed_corpus(args, corpus, model, tokenizer):
    """
    This function embeds all documents in the corpus given tokenizer and pre-trained encoder
    """
    model.eval()
    save_data = []
    C = tokenizer.batch_encode_plus(corpus, padding='max_length', truncation=True, max_length=args.ctx_len, return_tensors='pt')
    input_ids = C['input_ids'].to("cuda")
    attn_mask = C['attention_mask'].to("cuda")
    with torch.no_grad():
        for i in range(len(input_ids)):
            ex = T.tolist(model.get_representation(input_ids[i].view(1,-1), attn_mask[i].view(1,-1))[0])
            save_data.append(ex)
    return save_data