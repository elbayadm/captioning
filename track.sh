#! /usr/bin/zsh
# Run in edgar
track_myjobs_edgar(){
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
}

track_myjobs_lig(){
    jobs=("${(@f)$(myjobs | grep cap)}")
    for job in $jobs; do 
        ech $job
        #jobn=${${job%,T=*}#*N=}
        jid="$(cut -d' ' -f1 <<< $job)"
        jobn=$(oarstat -j  $jid -f | grep 'stderr_file')
        jobn="$(cut -d' ' -f7 <<< $jobn)"
        jobn="$(cut -d'/' -f2 <<< $jobn)"
        echo 'Job:' $jobn '('$jid')'
        python scripts/show.py -f $jobn
    done
}


case ${HOST:r:r} in 
    edgar) track_myjobs_edgar;;
    decore*) track_myjobs_lig;;
    dvorak*) track_myjobs_lig;;
    hyperion*) track_myjobs_lig;;
    *)  echo "Unknown whereabouts!!";;
esac


