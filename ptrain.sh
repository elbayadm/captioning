#! /usr/bin/bash
JOB=$1
HOST=$2

echo traininig $JOB
mkdir -p 'save/'$JOB
oarsub -l "walltime=24:0:0" -n $JOB \
       -t besteffort -t idempotent \
       -p "host='gpuhost$HOST'"\
       -O  save/$JOB/stdout -E save/$JOB/stderr\
       'python train.py -c config/'$JOB'.yaml'


