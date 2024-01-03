#!/bin/bash

outdir='./results/results_heart_disease_1226_1'
seed0=42

for rep in {0..4}
do  
    seed=`expr $rep + $seed0`
    echo "python heart_disease_rpdp_fed.py --outdir $outdir --seed $seed"
    nohup python -u heart_disease_rpdp_fed.py --outdir $outdir --seed $seed >logs/log_$seed 2>&1 & 
done
