import torch
from torch.utils.data import Dataset
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class RacyCodesDataset(Dataset):
    def __init__(self, source_codes, tokenizer):

        self.source_codes = source_codes
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.source_codes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() 
        file_name, source_code = self.source_codes[idx]
        tokenized_input = self.tokenizer.encode(source_code).ids
        tokenized_input = torch.tensor(tokenized_input, device=device, dtype=int)

        return file_name, tokenized_input