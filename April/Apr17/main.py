from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
import re
import torch
import torch.nn as nn

app = FastAPI()

vocab = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

def normalize_text(text):
    text = text.lower()
    new_text = ""
    for char in text:
        if char in vocab:
            new_text += char
        else:
            new_text += " "
    new_text = re.sub(r'\s+', ' ', new_text)
    return new_text

vocab_lookup = {}
for i in range(27):
    dictionary = {}
    vocab_lookup[vocab[i]] = dictionary
print(vocab_lookup)



app.mount('/', StaticFiles(directory='static', html=True), name='static')
