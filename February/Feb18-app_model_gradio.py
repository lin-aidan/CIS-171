import gradio as gr
import torch
import torch.nn as nn

model_data = torch.load('model.pth')
fm = model_data['fm']
fs = model_data['fs']
tm = model_data['tm']
ts = model_data['ts']
parameters = model_data['parameters']

def f(x):
    features = torch.tensor([[x]])
    X = (features - fm)/fs
    model = nn.Linear(1, 1) # 1 input, 1 output
    model.load_state_dict(parameters)
    prediction = model(X)
    return (prediction * ts + tm).item()

with gr.Blocks() as iface:
    x_box = gr.Number(label = 'Enter a square footage')
    price_box = gr.Number(label = 'This is the predicted price')
    x_box.change(fn = f, inputs = [x_box], outputs = [price_box])

iface.launch()