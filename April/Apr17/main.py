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

embedding = nn.Embedding(len(vocab), 5)
vocab_lookup = {}
for i, char in enumerate(vocab):
    dictionary = {
        "char": char,
        "index": i,
        "embedding": embedding(torch.tensor([i])).tolist()
    }
    vocab_lookup[char] = dictionary
print(vocab_lookup)



@app.get('/api/vocab')
def get_vocab_lookup():
    return vocab_lookup


@app.get('/myballs')
def my_balls():
    return ['lick', 'my', 'balls']


app.mount('/', StaticFiles(directory='static', html=True), name='static')
