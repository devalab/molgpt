import moses
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required = True, help="name of the generated dataset")
    args = parser.parse_args()

    data = pd.read_csv(args.path)

    test = moses.get_all_metrics(list(data['smiles'].values), device = 'cuda')

    print(path)
    print(test)
    print('*'*50)
