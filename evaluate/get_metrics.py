import pandas as pd
import argparse

from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import Chem
from utils import check_novelty, canonic_smiles

def calculate_un(results, moses):

    canon_smiles = [canonic_smiles(s) for s in results['smiles']]
    unique_smiles = list(set(canon_smiles))
    novel_ratio = check_novelty(unique_smiles, set(moses[moses['split']=='train']['smiles']))   # replace 'source' with 'split' for moses

    return len(unique_smiles)/len(results), novel_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--props', nargs="+", default = [], help="properties to be used for condition", required=True)
    parser.add_argument('--path', type=str, help="path of csv", required=True)

    args = parser.parse_args()


    data = pd.read_csv(args.path)
    moses = pd.read_csv('datasets/moses2.csv')
    moses = moses.dropna(axis=0).reset_index(drop=True)
    moses.columns = moses.columns.str.lower()


    if 'scaffold_cond' in data.columns:


        data['scaffold_cond'] = data['scaffold_cond'].apply(lambda x: x.replace('<',''))
        data['mol_scaf'] = data['smiles'].apply(lambda x: MurckoScaffoldSmiles(x))
        data['fp'] = data['mol_scaf'].apply(lambda x: FingerprintMols.FingerprintMol(Chem.MolFromSmiles(x)))
        data['cond_fp'] = data['scaffold_cond'].apply(lambda x: FingerprintMols.FingerprintMol(Chem.MolFromSmiles(x)))
        data['similarity'] = -1

        for idx, row in data.iterrows():
            data.loc[idx, 'similarity'] = TanimotoSimilarity(row['fp'], row['cond_fp'])

        # Fraction of valid molecules having tanimoto similarity of conditional scaffold and generated mol scaffold as 1
        x = data['scaffold_cond'].value_counts()
        y = data[data['similarity'] == 1]['scaffold_cond'].value_counts()
        print(y.divide(x))

        new_df = []
        for cond in data['scaffold_cond'].unique():
            scaffold_samples = len(data[data['scaffold_cond'] == cond].reset_index(drop = True))
            results = data[(data['scaffold_cond'] == cond) & (data['similarity'] > 0.8)].reset_index(drop = True)
            val = len(results) / scaffold_samples
            previous_validity = results['validity'][0]
            uniqueness, novelty = calculate_un(results, moses)
            results['validity'] = val * previous_validity
            results['unique'] = uniqueness
            results['novelty'] = novelty
            new_df.append(results)

        data = pd.concat(new_df).reset_index(drop = True)

        avg_validity = data.groupby('scaffold_cond')['validity'].mean()
        avg_unique = data.groupby('scaffold_cond')['unique'].mean()
        avg_novelty = data.groupby('scaffold_cond')['novelty'].mean()

        print('Validity \n')
        print(avg_validity)
        print('\n Uniqueness \n')
        print(avg_unique)
        print('\n Novelty \n')
        print(avg_novelty)

        if len(args.props) == 1:
            data['difference'] = abs(data['condition'] - data[args.props[0]])
            print(f'\n Mean Absolute Difference: {args.props[0]} \n')
            print(data.groupby('scaffold_cond')['difference'].mean())
            print(f'\n Standard Deviation of the Difference: {args.props[0]} \n')
            print(data.groupby('scaffold_cond')['difference'].std())
        else:
            for idx, p in enumerate(args.props):
                data[f'{p}_condition'] = data['condition'].apply(lambda x: tuple(float(s) for s in x.strip("()").split(","))[idx])

                data['difference'] = abs(data[f'{p}_condition'] - data[p])
                print(f'\n Mean Absolute Difference: {p} \n')
                print(data.groupby('scaffold_cond')['difference'].mean())
                print(f'\n Standard Deviation of the Difference: {p} \n')
                print(data.groupby('scaffold_cond')['difference'].std())


    else:

        avg_validity = data['validity'].mean()
        avg_unique = data['unique'].mean()
        avg_novelty = data['novelty'].mean()

        print('Validity \n')
        print(avg_validity)
        print('\n Uniqueness \n')
        print(avg_unique)
        print('\n Novelty \n')
        print(avg_novelty)

        if len(args.props) == 1:
            data['difference'] = abs(data['condition'] - data[args.props[0]])
            print(f'\n Mean Absolute Difference: {args.props[0]} \n')
            print(data['difference'].mean())
            print(f'\n Standard Deviation of the Difference: {args.props[0]} \n')
            print(data['difference'].std())
        else:
            for idx, p in enumerate(args.props):
                data[f'{p}_condition'] = data['condition'].apply(lambda x: tuple(float(s) for s in x.strip("()").split(","))[idx])

                data['difference'] = abs(data[f'{p}_condition'] - data[p])
                print(f'\n Mean Absolute Difference: {p} \n')
                print(data['difference'].mean())
                print(f'\n Standard Deviation of the Difference: {p} \n')
                print(data['difference'].std())
