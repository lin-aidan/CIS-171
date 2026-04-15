from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
import re
import torch
import torch.nn as nn

app = FastAPI()

vocab = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

vocab_dict = {}
for i, j in enumerate(vocab):
    vocab_dict[j] = i
print(vocab_dict)

@app.get('/vocab')
def get_vocab():
    return vocab_dict

app.mount('/', StaticFiles(directory='static', html=True), name='static')
