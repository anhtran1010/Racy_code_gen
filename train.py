from model.transformerVAE import TransformerVAE
from utils.kl_annealing import KLAnnealer
import json
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, KFold, ShuffleSplit
from tokenizers import Tokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = Tokenizer.from_file("/home/abtran/Racy_code_gen/utils/buggy_code_tokenizer.json")
tokenizer.enable_truncation(max_length=4096)

with open("/home/abtran/Racy_code_gen/racy_codes.json") as cf:
    code_files = json.load(cf)
    
rs = ShuffleSplit(n_splits=1, random_state=0, test_size=0.10, train_size=None)
model = TransformerVAE(embed_dim=64, src_vocab_size=tokenizer.get_vocab_size(),  seq_length=4096).to(device=torch.device(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

train_losses = []
test_losses = []

for epoch in range(50):
    print(f"starting epoch {epoch}")
    train_loss = []
    test_loss = []
    
    for train,test in rs.split(code_files):
        train_set = train
        test_set = test
    loss_fn = KLAnnealer(total_steps=len(train_set))
    model.train()
    for train_index in train_set:
        file_name, source_code = code_files[train_index]
        tokenized_input = tokenizer.encode(source_code).ids
        tokenized_input = torch.tensor(tokenized_input, device=device, dtype=int)
        output, mu, logvar = model(tokenized_input)
        loss = loss_fn(output, tokenized_input, logvar, mu)
        train_loss.append(loss.cpu().data.numpy())
        optimizer.zero_grad()
        loss.backward()
        grad_clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=100.0
        )
        optimizer.step()
    train_losses.append(np.mean(train_loss))
    
    model.eval()
    for test_index in test_set:
        file_name, source_code = code_files[train_index]
        tokenized_input = tokenizer.encode(source_code).ids
        tokenized_input = torch.tensor(tokenized_input, device=device, dtype=int)
        output, mu, logvar = model(tokenized_input)
        loss = loss_fn(output, tokenized_input, logvar, mu)
        test_loss.append(loss.cpu().data.numpy())
    test_losses.append(np.mean(test_loss))

plt.figure(figsize=(12, 8))
plt.plot(np.arange(50), train_losses, label="Train Loss",  color='blue')
plt.plot(np.arange(50), test_losses, label="Test Loss",  color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs Test Losses')
plt.savefig("Train_Test_losses")
    

