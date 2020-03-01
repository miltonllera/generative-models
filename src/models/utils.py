import os
import torch
import yaml


def load_meta(path):
    with open(path, mode='r') as f:
        meta = yaml.load(f, yaml.FullLoader)
    return meta


def from_file(model_folder, inner_loader, device='cuda'):
    for file in os.listdir(model_folder):
        if file.endswith((".p", ".pth")):
            file = os.path.join(model_folder, file)
            meta = os.path.join(model_folder, 'meta.yaml')
            
            meta = load_meta(meta)

            with open(file, mode='rb') as f:
                state_dict = torch.load(f)

            return inner_loader(meta['model-params'], state_dict, device), meta
