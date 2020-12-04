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

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--run_name', type=str, help="name for wandb run", required=False)
	parser.add_argument('--debug', action='store_true', default=False, help='debug')
	parser.add_argument('--data_name', type=str, default = 'moses', help="name of the dataset to train on", required=False)
	parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
	parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
	parser.add_argument('--prop1_unique', type=str, default = 0, help="unique values in that property", required=False)
	parser.add_argument('--n_layer', type=int, default = 8, help="number of layers", required=False)
	parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
	parser.add_argument('--n_embd', type=int, default = 256, help="embedding dimension", required=False)
	parser.add_argument('--max_epochs', type=int, default = 10, help="total epochs", required=False)
	parser.add_argument('--batch_size', type=int, default = 512, help="batch size", required=False)
	parser.add_argument('--learning_rate', type=int, default = 6e-4, help="learning rate", required=False)


	args = parser.parse_args()

	set_seed(42)

	wandb.init(project="lig_gpt", name=args.run_name)

	data = pd.read_csv('datasets/' + args.data_name + '.csv')
	data.columns = data.columns.str.lower()

	train_data = data[data['split']=='train'].reset_index(drop=True)
	val_data = data[data['split']=='test'].reset_index(drop=True)

	smiles = data['smiles']
	vsmiles = val_data['smiles']
	

	if args.property == 'logp':
		prop = data[args.property] + 6
		prop = np.array(pd.cut(prop.values, [0, 7, 8, 9, 10, 12], labels = [0, 1, 2, 3, 4]))

		vprop = val_data[args.property]
		prop = np.array(pd.cut(prop.values, [0, 7, 8, 9, 10, 12], labels = [0, 1, 2, 3, 4]))

	elif args.property == 'qed':
		prop = data[args.property]
		prop = np.array(pd.cut(prop.values, [0, 0.6, 0.7, 0.8, 0.9, 1.0], labels = [0, 1, 2, 3, 4]))

		vprop = val_data[args.property]
		vprop = np.array(pd.cut(vprop.values, [0, 0.6, 0.7, 0.8, 0.9, 1.0], labels = [0, 1, 2, 3, 4]))
	else:
		prop = data[args.property]
		prop = np.array(pd.cut(prop.values, [250, 270, 290, 310, 330, 350], labels = [0, 1, 2, 3, 4]))

		vprop = val_data[args.property]
		vprop = np.array(pd.cut(vprop.values, [250, 270, 290, 310, 330, 350], labels = [0, 1, 2, 3, 4]))


	lens = [len(i) for i in (list(smiles.values) + list(vsmiles.values))]
	max_len = max(lens)
	smiles = [ i + str('<')*(max_len - len(i)) for i in smiles]
	vsmiles = [ i + str('<')*(max_len - len(i)) for i in vsmiles]
	content = ' '.join(smiles + vsmiles)

	block_size = max_len
	print(block_size)

	train_dataset = SmileDataset(args, smiles, content, block_size, prop)
	valid_dataset = SmileDataset(args, vsmiles, content, block_size, prop)

	mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, num_props = args.num_props, prop1_embd = args.prop1_unique,
	               n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
	model = GPT(mconf)

	tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
	                      lr_decay=True, warmup_tokens=0.1*len(train_data)*block_size, final_tokens=args.max_epochs*len(train_data)*block_size,
	                      num_workers=10, ckpt_path = '../cond_gpt/weights/gpt_moses.pt')
	trainer = Trainer(model, train_dataset, valid_dataset, tconf)
	trainer.train(wandb)
