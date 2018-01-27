#! /usr/bin/zsh
# Run in edgar
jobs=("${(@f)$(oarstat | grep melbayad)}")
for job in $jobs; do 
    sj=${${job%,T=*}#*N=}
    echo $sj
    python scripts/show.py -f $sj
done
