# LigGPT
In this work, we train small custom GPT on Moses and Guacamol dataset with next token prediction task. The model is then used for unconditional and conditional molecular generation. We compare our model with previous approaches on the Moses and Guacamol datasets. Saliency maps are obtained for interpretability using Ecco library.

To train the model, make sure you have the datasets' csv file in the same directory as the code files.
For unconditional training run:

```python
python train.py --run_name unconditional_moses --data_name moses --num_props 0 
```

For property based conditional training:

```python
python train.py --run_name conditional_moses --data_name moses --num_props 1 --property logp
```

For scaffold based conditional training:

```python
python train.py --run_name scaffold_moses --data_name moses --scaffold --num_props 0
```

If you find this work useful, please cite:

Bagal, Viraj; Aggarwal, Rishal; Vinod, P. K.; Priyakumar, U. Deva (2021): LigGPT: Molecular Generation using a Transformer-Decoder Model. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.14561901.v1 
