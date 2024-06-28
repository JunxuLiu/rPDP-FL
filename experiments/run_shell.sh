#!/bin/bash
seed0=42
gpu0=3
for rep in {1..4}
do  
    seed=`expr $rep + $seed0`
    gpuid=`expr $rep + $gpu0`
    echo "python fedavg_rpdp.py --dataset snli --seed $seed --gpuid $gpuid"
    nohup python -u fedavg_rpdp.py --dataset snli --seed $seed --gpuid $gpuid >logs/log_snli_${seed}_${gpuid} 2>&1 & 
done