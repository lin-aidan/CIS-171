import torch
import torch.nn as nn
import gradio as gr

model_data = torch.load("Feb23-tumor.pth")
fm = model_data['fm']
fs = model_data['fs']
parameters = model_data['parameters']

linear = nn.Linear(1, 1)
linear.load_state_dict(parameters)
model = nn.Sequential(linear, nn.Sigmoid())

def f(x):
    features = torch.tensor([[x]])
    X = (features - fm)/fs
    prob = model(X) # probability gets returned, not binary classification yet
    if prob > .5:
        classification = "Malignant"
    else:
        classification = "Benign"
    return classification

with gr.Blocks() as iface:
    x_box = gr.Number(label = 'Enter tumor size')
    class_box = gr.Textbox(label = 'Classification')
    x_box.change(fn = f, inputs = [x_box], outputs = [class_box])

iface.launch()