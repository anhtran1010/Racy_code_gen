from model.transformerVAE import TransformerVAE
from utils.kl_annealing import KLAnnealer
from tokenizers import Tokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np
from code_dataset import RacyCodesDataset
from torch.utils.data import DataLoader, random_split
import json 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = Tokenizer.from_file("/home/abtran/Racy_code_gen/buggy_code_tokens.json")
tokenizer.enable_truncation(max_length=4096)

with open("race_codes.json") as rc:
    race_codes = json.load(rc)
model = torch.load('/home/abtran/Racy_code_gen/racecode_vae.pth').to(device=device)


mean = torch.zeros(4096, 8).to(device)
var = torch.ones(4096, 8).to(device)
epsilon = torch.randn_like(var).to(device)      
z_sample = mean + var*epsilon
x_decoded = model.decode(z_sample)
print(x_decoded.shape)
print(tokenizer.decode(torch.argmax(x_decoded, dim=1).tolist()))