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
model = TransformerVAE(embed_dim=64, src_vocab_size=tokenizer.get_vocab_size(),  seq_length=4096).to(device=torch.device(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
train_code, test_code = random_split(race_codes, [0.9, 0.1])
train_set = RacyCodesDataset(train_code, tokenizer)
test_set = RacyCodesDataset(test_code, tokenizer)
train_data = DataLoader(train_set)
test_data = DataLoader(test_set)
train_losses = []
test_losses = []

for epoch in range(50):
    print(f"starting epoch {epoch}")
    train_loss = []
    test_loss = []
    loss_fn = KLAnnealer(total_steps=len(train_data))
    model.train()
    for file_name, tokenized_input in train_data:
        output, mu, logvar = model(tokenized_input[0])
        loss = loss_fn(output, tokenized_input[0], logvar, mu)
        train_loss.append(loss.cpu().data.numpy())
        optimizer.zero_grad()
        loss.backward()
        grad_clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=100.0
        )
        optimizer.step()
    train_losses.append(np.mean(train_loss))
    
    model.eval()
    for file_name, tokenized_input in test_data:
        output, mu, logvar = model(tokenized_input)
        loss = loss_fn(output, tokenized_input, logvar, mu)
        test_loss.append(loss.cpu().data.numpy())
    test_losses.append(np.mean(test_loss))
    
torch.save(model, 'racecode_vae.pth')
plt.figure(figsize=(12, 8))
plt.plot(np.arange(50), train_losses, label="Train Loss",  color='blue')
plt.plot(np.arange(50), test_losses, label="Test Loss",  color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs Test Losses')
plt.savefig("Train_Test_losses")
    

