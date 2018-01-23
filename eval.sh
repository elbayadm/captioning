#! /usr/bin/bash
model1=$1

oarsub -l "walltime=24:0:0" -n eval_$1 \
       -t besteffort -t idempotent \
       -p "not gpumodel='k40m' and gpumem>6000" \
       -O  eval_stdout -E eval_stderr \
       'python eval.py -c config/'$model1'.yaml --split test'




