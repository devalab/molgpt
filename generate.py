from utils import check_novelty, sample, canonic_smiles
from dataset import SmileDataset
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import math
from tqdm import tqdm
import argparse
from model import GPT, GPTConfig
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from moses.utils import get_mol
import re
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
	parser.add_argument('--scaffold', action='store_true', default=False, help='condition on scaffold')
	parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
	parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=True)
	parser.add_argument('--data_name', type=str, default = 'moses2', help="name of the dataset to train on", required=False)
	parser.add_argument('--gen_size', type=int, default = 10000, help="number of times to generate from a batch", required=False)
	parser.add_argument('--vocab_size', type=int, default = 26, help="number of layers", required=False)  # previously 28
	parser.add_argument('--block_size', type=int, default = 54, help="number of layers", required=False)   # previously 57
	parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
	parser.add_argument('--n_layer', type=int, default = 8, help="number of layers", required=False)
	parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
	parser.add_argument('--n_embd', type=int, default = 256, help="embedding dimension", required=False)
	parser.add_argument('--lstm_layers', type=int, default = 2, help="number of layers in lstm", required=False)

	args = parser.parse_args()

	
	context = "C"

	label2molwt = {0: 260, 1: 280, 2: 300, 3: 320, 4: 340}
	label2qed = {0: '0-0.6', 1: '0.6-0.7', 2: '0.7-0.8', 3: '0.8-0.9', 4: '0.9-1.0'}
	label2logp = {0: '-6-1', 1: '1-2', 2: '2-3', 3: '3-4', 4: '4-6'}


	data = pd.read_csv('datasets/' + args.data_name + '.csv')
	data = data.dropna(axis=0).reset_index(drop=True)
	data.columns = data.columns.str.lower()
	smiles = data[data['split']!='test_scaffolds']['smiles']
	scaf = data[data['split']!='test_scaffolds']['scaffold_smiles']

	# scaffold = data[data['split']!='test_scaffolds']['scaffold_smiles']
	# lens = [len(i.strip()) for i in scaffold.values]
	# scaffold_max_len = max(lens)

	# scaffold = data[data['split']=='test_scaffolds']['scaffold_smiles'].values
	# scaffold = sorted(list(scaffold))
	# condition = [scaffold[0], scaffold[len(scaffold)//2], scaffold[-1]]
	# condition = np.random.choice(scaffold, size = 3, replace = False)
	
	# condition = ['c1cnc2[nH]ccc2c1']
	# condition = ['O=C(CCc1cn[nH]c1)NCCC1CC2CCC1C2', 'O=C(CCC(=O)NCC1CCCO1)NCc1ccccc1', 'O=S(=O)(Cc1ccon1)NCc1cccs1']  # sim 0.9, 0.8, ~0.7
	

	pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
	regex = re.compile(pattern)
	lens = [len(regex.findall(i)) for i in smiles]
	max_len = max(lens)
	smiles = [ i + str('<')*(max_len - len(regex.findall(i))) for i in smiles]

	lens = [len(regex.findall(i)) for i in scaf]
	scaffold_max_len = max(lens)
	scaf = [ i + str('<')*(scaffold_max_len - len(regex.findall(i))) for i in scaf]
	
	content = ' '.join(smiles + scaf) 
	chars = sorted(list(set(regex.findall(content))))

	stoi = { ch:i for i,ch in enumerate(chars) }
	itos = { i:ch for i,ch in enumerate(chars) }

	condition = ['O=C(Cc1ccccc1)NCc1ccccc1', 'c1cnc2[nH]ccc2c1', 'c1ccc(-c2ccnnc2)cc1']
	condition = [ i + str('<')*(scaffold_max_len - len(regex.findall(i))) for i in condition]
	print(condition)

	mconf = GPTConfig(args.vocab_size, args.block_size, num_props = args.num_props,
	               n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold = args.scaffold, scaffold_maxlen = scaffold_max_len,
	               lstm = args.lstm, lstm_layers = args.lstm_layers)
	model = GPT(mconf)


	model.load_state_dict(torch.load('weights/' + args.model_weight))
	model.to('cuda')
	print('Model loaded')

	gen_iter = math.ceil(args.gen_size / 512)


	all_dfs = []
	# for j in [0.3, 0.5, 0.7, 0.9]:
	# for j in [[0.5, 0], [0.5, 4], [0.9, 0], [0.9, 4]]:
	# for j in condition:
		# for c in [1.0, 2.0, 3.0]:
	molecules = []
	for i in tqdm(range(gen_iter)):
	    x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(512, 1).to('cuda')
	    p = None
	    # p = torch.tensor([[c]]).repeat(512, 1).to('cuda')   # for single condition
	    # p = torch.tensor([j]).repeat(512, 1).unsqueeze(1).to('cuda')    # for multiple conditions
	    # sca = torch.tensor([stoi[s] for s in regex.findall(j)], dtype=torch.long)[None,...].repeat(512, 1).to('cuda')
	    sca = None
	    y = sample(model, x, args.block_size, temperature=1.6, sample=True, top_k=None, prop = p, scaffold = sca)
	    for gen_mol in y:
	        completion = ''.join([itos[int(i)] for i in gen_mol])
	        completion = completion.replace('<', '')
	        mol = get_mol(completion)
	        if mol:
	            molecules.append(mol)
        
	"Valid molecules % = {}".format(len(molecules))

	mol_dict = []

	for i in molecules:
		mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})


	results = pd.DataFrame(mol_dict)

	canon_smiles = [canonic_smiles(s) for s in results['smiles']]
	unique_smiles = list(set(canon_smiles))
	novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))

	# print(f'Condition: {j}')
	print('Valid ratio: ', np.round(len(results)/(512*gen_iter), 3))
	print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
	print('Novelty ratio: ', np.round(novel_ratio/100, 3))

	# results['condition'] = str((j[0], j[1]))
	# results['condition'] = c
	# results['scaffold_cond'] = j
	results['qed'] = results['molecule'].apply(lambda x: QED.qed(x) )
	results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x) )
		# all_dfs.append(results)


	# results = pd.concat(all_dfs)
	results.to_csv('gen_csv/' + args.csv_name + '.csv', index = False)

	unique_smiles = list(set(results['smiles']))
	canon_smiles = [canonic_smiles(s) for s in results['smiles']]
	unique_smiles = list(set(canon_smiles))
	novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))

	print('Valid ratio: ', np.round(len(results)/(512*gen_iter), 3))
	print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
	print('Novelty ratio: ', np.round(novel_ratio/100, 3))
