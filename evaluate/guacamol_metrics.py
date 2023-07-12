from guacamol.utils.chemistry import canonicalize_list, is_valid, calculate_pc_descriptors, continuous_kldiv, \
    discrete_kldiv, calculate_internal_pairwise_similarities
from utils import canonic_smiles
from fcd_torch import FCD as FCDMetric
import pandas as pd
import numpy as np
from moses.metrics.metrics import compute_intermediate_statistics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required = True, help="path to generated csv")
    args = parser.parse_args()

    gen = pd.read_csv(args.path)
    print(gen.shape)
    data = pd.read_csv('datasets/guacamol2.csv')
    data.columns = data.columns.str.lower()
    data = data.sample(n = 10000, random_state = 42).reset_index(drop = True)
    gen = gen.sample(n = 10000, random_state = 42).reset_index(drop = True)

    pc_descriptor_subset = [
        'BertzCT',
        'MolLogP',
        'MolWt',
        'TPSA',
        'NumHAcceptors',
        'NumHDonors',
        'NumRotatableBonds',
        'NumAliphaticRings',
        'NumAromaticRings'
    ]

    canon_smiles = [canonic_smiles(s) for s in gen['smiles']]
    unique_smiles = set(canon_smiles)
    training_set_molecules = list(data['smiles'].values)

    print('Calculating FCD')


    ptest = compute_intermediate_statistics(training_set_molecules, n_jobs=10,
                                            device='cuda',
                                            batch_size=512)

    fcd = FCDMetric(n_jobs = 10, device = 'cuda', batch_size = 512)(gen=list(gen['smiles'].values), pref=ptest['FCD'])


    print('FCD score: ', fcd)

    print('Calculating PC descriptors')
    d_sampled = calculate_pc_descriptors(unique_smiles, pc_descriptor_subset)
    d_chembl = calculate_pc_descriptors(training_set_molecules, pc_descriptor_subset)

    kldivs = {}

    # now we calculate the kl divergence for the float valued descriptors ...
    for i in range(4):
        print(f'Calculating {pc_descriptor_subset[i]}')
        kldiv = continuous_kldiv(X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i])
        kldivs[pc_descriptor_subset[i]] = kldiv

    # ... and for the int valued ones.
    for i in range(4, 9):
        print(f'Calculating {pc_descriptor_subset[i]}')
        kldiv = discrete_kldiv(X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i])
        kldivs[pc_descriptor_subset[i]] = kldiv

    # pairwise similarity
    print('Calculating pairwise similarity of training set')
    chembl_sim = calculate_internal_pairwise_similarities(training_set_molecules)
    chembl_sim = chembl_sim.max(axis=1)

    print('Calculating pairwise similarity of generated set')
    sampled_sim = calculate_internal_pairwise_similarities(unique_smiles)
    sampled_sim = sampled_sim.max(axis=1)

    print('Calculating KLDiv of similarity')
    kldiv_int_int = continuous_kldiv(X_baseline=chembl_sim, X_sampled=sampled_sim)
    kldivs['internal_similarity'] = kldiv_int_int

    partial_scores = [np.exp(-score) for score in kldivs.values()]
    score = sum(partial_scores) / len(partial_scores)

    print('KL Divergence: ', score)

