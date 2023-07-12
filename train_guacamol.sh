#!/bin/bash

#unconditional train/training
python train/train.py --run_name unconditional_guacamol --data_name guacamol2 --batch_size 384 --num_props 0 --max_epochs 10

# property based conditional train/training
python train/train.py --run_name logp_guacamol --data_name guacamol2 --batch_size 384 --num_props 1 --max_epochs 10 --props logp

python train/train.py --run_name logp_sas_guacamol --data_name guacamol2 --batch_size 384 --num_props 2 --max_epochs 10 --props logp sas 

# scaffold based conditional train/training
python train/train.py --run_name scaffold_guacamol --data_name guacamol2 --scaffold --batch_size 384 --max_epochs 10

# scaffold + property based conditional train/training
python train/train.py --run_name logp_scaffold_guacamol --data_name guacamol2 --scaffold --batch_size 384 --num_props 1 --max_epochs 10 --props logp

python train/train.py --run_name logp_sas_scaffold_guacamol --data_name guacamol2 --scaffold --batch_size 384 --num_props 2 --max_epochs 10 --props logp sas
