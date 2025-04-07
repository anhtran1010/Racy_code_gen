from model.transformerVAE import TransformerVAE
from utils.kl_annealing import KLAnnealer
from tokenizers import Tokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np
from code_dataset import RacyCodesDataset
from torch.utils.data import DataLoader, random_split
import json 
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = Tokenizer.from_file("/home/abtran/Racy_code_gen/buggy_code_tokens.json")
tokenizer.enable_truncation(max_length=4096)

with open("race_codes.json") as rc:
    race_codes = json.load(rc)
model = torch.load('/home/abtran/Racy_code_gen/racecode_vae.pth').to(device=device)


# mean = torch.zeros(4096, 8).to(device)
# var = torch.ones(4096, 8).to(device)
# epsilon = torch.randn_like(var).to(device)      
# z_sample = mean + var*epsilon
# x_decoded = model.decode(z_sample)
# print(x_decoded.shape)
# print(tokenizer.decode(torch.argmax(x_decoded, dim=1).tolist()))
with open("dataset/DRB028-privatemissing-orig-yes.c", "r") as co:
    code_one = co.read()

with open("dataset/DRB022-reductionmissing-var-yes.c", "r") as ct:
    code_two = ct.read()

code_one_tokens = torch.tensor(tokenizer.encode(code_one).ids, device=device, dtype=int)
code_two_tokens = torch.tensor(tokenizer.encode(code_two).ids, device=device, dtype=int)
if code_one_tokens.size(0) > code_two_tokens.size(0):
    pad_len = code_one_tokens.size(0) - code_two_tokens.size(0)
    enc_two_padded = F.pad(code_two_tokens, (0, pad_len))  # Pad at the end
    enc_one_padded = code_one_tokens
else:
    pad_len = code_two_tokens.size(0) - code_one_tokens.size(0)
    enc_one_padded = F.pad(code_one_tokens, (0, pad_len))
    enc_two_padded = code_two_tokens

mean_one, log_one = model.encoder(enc_one_padded)
enc_one = model.reparameterization(mean_one, log_one)
mean_two, log_two = model.encoder(enc_two_padded)
enc_two = model.reparameterization(mean_two, log_two)

print(enc_one.shape, enc_two.shape)
dist_vec = enc_two - enc_one
samples = torch.linspace(0,1, 5)

for s in samples:
    new_samp = enc_one + s * dist_vec
    print(new_samp.shape)
    new_code = model.decode(new_samp)
    print(tokenizer.decode(torch.argmax(new_code, dim=1).tolist()))

