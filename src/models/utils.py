import os
import torch
import yaml


def load_model(model_folder, inner_loader, device='cuda'):
    for file in os.listdir(model_folder):
        if file.endswith((".p", ".pth")):
            with open(os.path.join(model_folder, 'meta.yaml'), mode='r') as f:
                meta = yaml.load(f, yaml.FullLoader)
            with open(os.path.join(model_folder, file), mode='rb') as f:
                state_dict = torch.load(f)
            return inner_loader(meta['model-params'], state, device)
