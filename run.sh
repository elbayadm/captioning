#! /usr/bin/bash
echo "Submitting as besteffort Eval job"
oarsub -l "walltime=24:0:0" -n "Ensemble_eval_val2014" \
       -t besteffort -t idempotent \
       -p "gpumem > 6000"\
       -O  eval_stdout -E eval_stderr\
       'python eval_ensemble.py --model fn_baseline_resnet50_genconf15_lr1 fn_raml_exp_tau005_isolated_a03_genconf15_lr1 fn_raml_cider_a03_lr1 --verbose 1 --dump_json 1 --output_json ensemble_val2014.json --beam_size 3 --batch_size 1 --image_folder data/coco/images/val2014 --image_list data/coco/images/val.txt'

