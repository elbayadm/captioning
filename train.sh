#! /usr/bin/env bash

BQ=''
TX=''
K40=''
while getopts 'bmlk' flag; do
    echo "Reading flag "$flag
    case "${flag}" in
        b) BQ='true' ;;  # Besteffort
        m) MED='true' ;;  # Exclude gtx
        l) P100='true' ;;  # Request a p100
        k) K40='true' ;;  # tolerate k40
        *) error "Unexpected option ${flag}" ;;
    esac
done
shift $((OPTIND-1))
JOB=$1
JOBSH=${JOB:0:50}
echo traininig $JOB
mkdir -p 'save/'$JOB

if [ $K40 ]; then
    oarprop="\"gpumem>'4000'\""
else
    oarprop="\"gpumodel<>'k40m'\""
fi
if [ $MED ]; then
    oarprop="\"(gpumodel<>'k40m' and gpumodel<>'gtx1080' and gpumodel<>'gtx1080_ti')\""
fi
if  [ $P100 ]; then
    oarprop="\"gpumodel='p100'\""
fi
echo "OAR requirements:" $oarprop 

if [ $BQ ]; then
    echo "Submitting to besteffort queue"
    cmd="oarsub -l \"walltime=24:0:0\" -n $JOBSH \
           -t besteffort -t idempotent \
           -p $oarprop \
           -O  save/$JOB/stdout -E save/$JOB/stderr\
           'source activate safe && python train.py -c config/'$JOB'.yaml'"
    echo 'Running' $cmd
    eval $cmd
else
    echo "Submitting to default queue"
    cmd="oarsub -l \"walltime=24:0:0\" -n $JOBSH \
            -O  save/$JOB/stdout -E save/$JOB/stderr\
            -p $oarprop\
            'python train.py -c config/'$JOB'.yaml'"
    eval $cmd
fi


