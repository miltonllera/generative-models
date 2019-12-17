import os
import torch
import yaml


def load_meta(path):
    with open(path, mode='r') as f:
        meta = yaml.load(f, yaml.FullLoader)
    return meta


def load_model(model_folder, inner_loader, device='cuda'):
    for file in os.listdir(model_folder):
        if file.endswith((".p", ".pth")):
            file = os.path.join(model_folder, file)
            meta = os.path.join(model_folder, 'meta.yaml')
            return inner_loader(meta, file, device)