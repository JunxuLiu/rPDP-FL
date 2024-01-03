#!/bin/bash

outdir='./results/results_mnist_1225'
seed0=42

for rep in {0..4}
do  
    seed=`expr $rep + $seed0`
    echo "python mnist_rpdp_fed_test_nm.py --outdir $outdir --seed $seed"
    nohup python -u mnist_rpdp_fed_test_nm.py --outdir $outdir --seed $seed >logs/log_$seed 2>&1 & 
done
