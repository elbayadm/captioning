#! /usr/bin/bash
model1=$1

oarsub -l "walltime=1:0:0" -n eval_$1 \
       -t besteffort -t idempotent \
       -p "gpumem>6000" \
       -O  eval_stdout -E eval_stderr \
       'python eval.py -c config/'$model1'.yaml --split test'




