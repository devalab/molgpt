#!/bin/bash

# Single property and scaffold based generation
python generate/generate.py --model_weight moses_scaf_wholeseq_tpsa.pt --props tpsa --scaffold --data_name moses2 --csv_name moses_scaf_tpsa_temp1 --gen_size 10000 --batch_size 512
python generate/generate.py --model_weight moses_scaf_wholeseq_sas.pt --props sas --scaffold --data_name moses2 --csv_name moses_scaf_sas_temp1 --gen_size 10000 --batch_size 512
python generate/generate.py --model_weight moses_scaf_wholeseq_logp_newtokens.pt --props logp --scaffold --data_name moses2 --csv_name moses_scaf_logp_temp1 --gen_size 10000 --batch_size 512
python generate/generate.py --model_weight moses_scaf_wholeseq_qed.pt --props qed --scaffold --data_name moses2 --csv_name moses_scaf_qed_temp1 --gen_size 10000 --batch_size 512


# Two property and scaffold based generation
python generate/generate.py --model_weight moses_scaf_wholeseq_tpsa_logp.pt --props tpsa logp --scaffold --data_name moses2 --csv_name moses_scaf_tpsa_logp_temp1 --gen_size 10000 --batch_size 512
python generate/generate.py --model_weight moses_scaf_wholeseq_tpsa_sas.pt --props tpsa sas --scaffold --data_name moses2 --csv_name moses_scaf_tpsa_sas_temp1 --gen_size 10000 --batch_size 512
python generate/generate.py --model_weight moses_scaf_wholeseq_sas_logp.pt --props sas logp --scaffold --data_name moses2 --csv_name moses_scaf_sas_logp_temp1 --gen_size 10000 --batch_size 512

# Triple property and scaffold based generation
python generate/generate.py --model_weight moses_scaf_wholeseq_tpsa_logp_sas.pt --props tpsa logp sas --scaffold --data_name moses2 --csv_name moses_scaf_tpsa_logp_sas_temp1 --gen_size 10000 --batch_size 512
