#! /usr/bin/zsh
# Run in edgar
jobs=("${(@f)$(oarstat | grep melbayad)}")
for job in $jobs; do 
    #jobn=${${job%,T=*}#*N=}
    jid="$(cut -d' ' -f1 <<< $job)"
    jobn=$(oarstat -j  $jid -f | grep 'stderr_file')
    jobn="$(cut -d' ' -f7 <<< $jobn)"
    jobn="$(cut -d'/' -f2 <<< $jobn)"
    echo 'Job:' $jobn '('$jid')'
    python scripts/show.py -f $jobn
done
