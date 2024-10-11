import torch
import numpy as np

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from myapp.model import Model
from torchvision.transforms import ToTensor, Compose, Normalize

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# load model
model = Model()
model.load_state_dict(torch.load('myapp/model.ckpt'))

# app
with open('emnist-balanced-mapping.txt', 'r') as file:
    content = file.read()
lines = content.strip().split('\n')
labels_dict = {int(lines[i].split(' ')[0]): chr(int(lines[i].split(' ')[1])) for i in range(len(lines))}

# app
app = FastAPI(title='Symbol detection', docs_url='/docs')

transform = Compose([
    Normalize([0.5], [0.5])
])

# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):
    image = torch.tensor(list(map(float, image[1:-1].split(',')))).reshape((28, 28)).transpose(1, 0).unsqueeze(0).unsqueeze(0)
    x = transform(image)
    model.eval()
    with torch.no_grad():
        pred = model(x)
    return {'prediction': labels_dict[pred.argmax().item()]}

# static files
app.mount('/', StaticFiles(directory='static', html=True), name='static')
