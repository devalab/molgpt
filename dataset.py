import torch
from torch.utils.data import Dataset

class SmileDataset(Dataset):

    def __init__(self, args, data, content, block_size, prop = None):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.debug = args.debug
    
    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.block_size + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        smiles, prop = self.data[idx], self.prop[idx]
        len_smiles = len(smiles)
        dix =  [self.stoi[s] for s in smiles]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        prop = torch.tensor([prop], dtype=torch.long)
        return x, y, prop