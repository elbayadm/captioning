#! /usr/bin/bash
model1=$1

oarsub -l "walltime=3:0:0" -n eval_cs_$1 \
       -t besteffort -t idempotent \
       -p "gpumem>6000" \
       -O  eval_stdout -E eval_stderr \
       'python eval_cs.py -c config/'$model1'.yaml --split test && python eval_cs.py -c config/'$model1'.yaml --split val'




