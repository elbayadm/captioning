#! /usr/bin/bash
BQ=""
TX=""

while getopts 'b:t' flag; do
    echo "Reading flag "$flag
    case "${flag}" in
        b) BQ='true' ;;
        t) TX='true' ;;
        *) error "Unexpected option ${flag}" ;;
    esac
done
shift $((OPTIND-2))
JOB=$1
MEM=$2

echo traininig $JOB
mkdir -p 'save/'$JOB

if [ $BQ ] && [ $TX ]; then
    echo "Submitting as besteffort to titan_x"
    oarsub -l "walltime=50:0:0" -n $JOB \
           -t besteffort -t idempotent \
           -p "gpumodel='titan_x' or gpumodel='titan_x_pascal'"\
           -O  save/$JOB/stdout -E save/$JOB/stderr\
           'python train.py -c config/'$JOB'.yaml'
else
    if [ $BQ ]; then
        echo "Submitting as besteffort"
        oarsub -l "walltime=50:0:0" -n $JOB \
            -t besteffort -t idempotent \
            -p "gpumem>"$MEM \
            -O  save/$JOB/stdout -E save/$JOB/stderr\
            'python train.py -c config/'$JOB'.yaml'
    else
        if [ $TX ]; then
            echo "Requesting a titan_x (high priority)"
            oarsub -l "walltime=50:0:0" -n $JOB \
                -O  save/$JOB/stdout -E save/$JOB/stderr\
                -p "gpumodel='titan_x' or gpumodel='titan_x_pascal'"\
                'python train.py -c config/'$JOB'.yaml'
        else
            echo "Standard (high priority)"
            oarsub -l "walltime=50:0:0" -n $JOB \
                -O  save/$JOB/stdout -E save/$JOB/stderr\
                -p "gpumem>"$MEM\
                'python train.py -c config/'$JOB'.yaml'
        fi
    fi
fi

