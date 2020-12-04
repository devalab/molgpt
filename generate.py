from utils import check_novelty, sample
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

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
	parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=True)
	parser.add_argument('--data_name', type=str, default = 'moses', help="name of the dataset to train on", required=False)
	parser.add_argument('--gen_size', type=int, default = 1000, help="number of times to generate from a batch", required=False)
	parser.add_argument('--vocab_size', type=int, default = 28, help="number of layers", required=False)
	parser.add_argument('--block_size', type=int, default = 57, help="number of layers", required=False)
	parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
	parser.add_argument('--prop1_unique', type=str, default = 0, help="unique values in that property", required=False)
	parser.add_argument('--n_layer', type=int, default = 8, help="number of layers", required=False)
	parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
	parser.add_argument('--n_embd', type=int, default = 256, help="embedding dimension", required=False)

	args = parser.parse_args()

	molecules = []
	context = "C"

	label2molwt = {0: 260, 1: 280, 2: 300, 3: 320, 4: 340}
	label2qed = {0: '0-0.6', 1: '0.6-0.7', 2: '0.7-0.8', 3: '0.8-0.9', 4: '0.9-1.0'}
	label2logp = {0: '-6-1', 1: '1-2', 2: '2-3', 3: '3-4', 4: '4-6'}

	mconf = GPTConfig(args.vocab_size, args.block_size, num_props = args.num_props, prop1_embd = args.prop1_unique,
	               n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
	model = GPT(mconf)

	data = pd.read_csv('datasets/' + args.data_name + '.csv')
	data.columns = data.columns.str.lower()
	smiles = data['smiles']

	lens = [len(i) for i in smiles]
	max_len = max(lens)
	smiles = [ i + str('<')*(max_len - len(i)) for i in smiles]
	content = ' '.join(smiles)
	chars = sorted(list(set(content)))

	stoi = { ch:i for i,ch in enumerate(chars) }
	itos = { i:ch for i,ch in enumerate(chars) }


	model.load_state_dict(torch.load('weights/' + args.model_weight))
	model.to('cuda')
	print('Model loaded')

	gen_iter = math.ceil(args.gen_size / 512)

	for i in tqdm(range(gen_iter)):
	    x = torch.tensor([stoi[s] for s in context], dtype=torch.long)[None,...].repeat(512, 1).to('cuda')
	    p = None
	    y = sample(model, x, args.block_size, temperature=0.9, sample=True, top_k=5, prop = p)
	    for gen_mol in y:
	        completion = ''.join([itos[int(i)] for i in gen_mol])
	        completion = completion.replace('<', '')
	        mol = Chem.MolFromSmiles(completion)
	        if mol:
	            molecules.append(mol)
	        
	"Valid molecules % = {}".format(len(molecules))


	mol_dict = []


	# for i, wt in molecules:
	#     mol_dict.append({'molecule' : i, 'molwt': ExactMolWt(i), 'smiles': Chem.MolToSmiles(i), 'cond_molwt': wt})
	    
	# for i, qe in molecules:
	#     mol_dict.append({'molecule' : i, 'qed': qed(i), 'smiles': Chem.MolToSmiles(i), 'cond_qed': qe})

	# for i, logp in molecules:
	#     mol_dict.append({'molecule' : i, 'logp': Crippen.MolLogP(i), 'smiles': Chem.MolToSmiles(i), 'cond_logp': logp})
	    
	for i in molecules:
	    mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})

	results = pd.DataFrame(mol_dict)
	results.to_csv('gen_csv/' + args.csv_name + '.csv', index = False)

	unique_smiles = list(set(results['smiles']))
	novel_ratio = check_novelty(unique_smiles, data['smiles'])

	print('Valid ratio: ', np.round(len(results)/(512*gen_iter), 3))
	print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
	print('Novelty ratio: ', np.round(novel_ratio/100, 3))