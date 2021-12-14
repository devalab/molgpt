#!/bin/bash

# Single property
python generate/generate.py --model_weight gua_tpsa.pt --props tpsa --data_name guacamol2 --csv_name gua_tpsa_temp1 --gen_size 10000 --batch_size 512 --vocab_size 94 --block_size 100
python generate/generate.py --model_weight guacamol_sas.pt --props sas --data_name guacamol2 --csv_name gua_sas_temp1 --gen_size 10000 --batch_size 512 --vocab_size 94 --block_size 100
python generate/generate.py --model_weight guacamol_logp.pt --props logp --data_name guacamol2 --csv_name gua_logp_temp1 --gen_size 10000 --batch_size 512 --vocab_size 94 --block_size 100
python generate/generate.py --model_weight guacamol_qed.pt --props qed --data_name guacamol2 --csv_name gua_qed_temp1 --gen_size 10000 --batch_size 512 --vocab_size 94 --block_size 100


# Two property generation
python generate/generate.py --model_weight gua_tpsa_logp.pt --props tpsa logp --data_name guacamol2 --csv_name gua_tpsa_logp_temp1 --gen_size 10000 --batch_size 512 --vocab_size 94 --block_size 100
python generate/generate.py --model_weight gua_tpsa_sas.pt --props tpsa sas --data_name guacamol2 --csv_name gua_tpsa_sas_temp1 --gen_size 10000 --batch_size 512 --vocab_size 94 --block_size 100
python generate/generate.py --model_weight guacamol_sas_logp.pt --props sas logp --data_name guacamol2 --csv_name gua_sas_logp_temp1 --gen_size 10000 --batch_size 512 --vocab_size 94 --block_size 100

# Triple property generation
python generate/generate.py --model_weight gua_tpsa_logp_sas.pt --props tpsa logp sas --data_name guacamol2 --csv_name gua_tpsa_logp_sas_temp1 --gen_size 10000 --batch_size 512 --vocab_size 94 --block_size 100
