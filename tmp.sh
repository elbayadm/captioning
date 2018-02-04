# /usr/bin/env zsh

for f in $(ls save | grep fncnn6_reset);
do 
    echo $f && python eval.py -c config/$f.yaml --split test ;
done
