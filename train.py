import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--run_name', type=str, help="name for wandb run", required=False)
	parser.add_argument('--debug', action='store_true', default=False, help='debug')
	parser.add_argument('--scaffold', action='store_true', default=False, help='condition on scaffold') # in moses dataset, on average, there are only 5 molecules per scaffold
	parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
	parser.add_argument('--data_name', type=str, default = 'moses2', help="name of the dataset to train on", required=False)
	parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
	parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
	parser.add_argument('--prop1_unique', type=int, default = 0, help="unique values in that property", required=False)
	parser.add_argument('--n_layer', type=int, default = 8, help="number of layers", required=False)
	parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
	parser.add_argument('--n_embd', type=int, default = 256, help="embedding dimension", required=False)
	parser.add_argument('--max_epochs', type=int, default = 10, help="total epochs", required=False)
	parser.add_argument('--batch_size', type=int, default = 512, help="batch size", required=False)
	parser.add_argument('--learning_rate', type=int, default = 6e-4, help="learning rate", required=False)
	parser.add_argument('--lstm_layers', type=int, default = 2, help="number of layers in lstm", required=False)

	args = parser.parse_args()

	set_seed(42)

	wandb.init(project="lig_gpt", name=args.run_name)

	data = pd.read_csv('datasets/' + args.data_name + '.csv')
	data = data.dropna(axis=0).reset_index(drop=True)
	data.columns = data.columns.str.lower()

	train_data = data[data['split']=='train'].reset_index(drop=True)
	val_data = data[data['split']=='test'].reset_index(drop=True)

	smiles = train_data['smiles']
	vsmiles = val_data['smiles']
	
	# prop = train_data[['qed', 'logp']]
	# vprop = val_data[['qed', 'logp']]

	prop = train_data['logp']
	vprop = val_data['logp']

	scaffold = train_data['scaffold_smiles']
	vscaffold = val_data['scaffold_smiles']



	lens = [len(i.strip()) for i in (list(smiles.values) + list(vsmiles.values))]
	max_len = max(lens)

	lens = [len(i.strip()) for i in (list(scaffold.values) + list(vscaffold.values))]
	scaffold_max_len = max(lens)

	smiles = [ i + str('<')*(max_len - len(i)) for i in smiles]
	vsmiles = [ i + str('<')*(max_len - len(i)) for i in vsmiles]

	scaffold = [ i + str('<')*(scaffold_max_len - len(i)) for i in scaffold]
	vscaffold = [ i + str('<')*(scaffold_max_len - len(i)) for i in vscaffold]

	whole_string = ' '.join(smiles + vsmiles)
	whole_string = sorted(list(set(whole_string)))
	print(whole_string)


	train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop = prop, aug_prob = 0, scaffold = scaffold, scaffold_maxlen = scaffold_max_len)
	valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop = vprop, aug_prob = 0, scaffold = vscaffold, scaffold_maxlen = scaffold_max_len)

	mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props = args.num_props,
	               n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold = args.scaffold, scaffold_maxlen = scaffold_max_len,
	               lstm = args.lstm, lstm_layers = args.lstm_layers)
	model = GPT(mconf)

	tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
	                      lr_decay=True, warmup_tokens=0.1*len(train_data)*max_len, final_tokens=args.max_epochs*len(train_data)*max_len,
	                      num_workers=10, ckpt_path = '../cond_gpt/weights/moses_scaf_lstm.pt')
	trainer = Trainer(model, train_dataset, valid_dataset, tconf)
	trainer.train(wandb)
