if [ $# -lt 14 ]
then
    echo usage:
    echo ./launcher.sh "<lower beta>" "<higher beta>" "<beta stepsize>" "<lower h>" "<higher h>" "<h stepsize>"  "<num iterations>" "<restart steps>" "<ene qbits>" "<stem outfile>" "<max reverse>" "<PE time>" "<PE steps>" "<stem X matrix>" "<reverse_count>"
else
    betal=$1
    betah=$2
    betas_stepsize=$3
    hl=$4
    hh=$5
    hs_stepsize=$6
    num_iters=$7
    restart_num=$8
    ne=$9
    outfile_stem=${10}
    max_reverse=${11}
    pe_time=${12}
    pe_steps=${13}
    X_mat_stem=${14}
    reverse_count=${15}
    for i in $(LC_NUMERIC=en_US.UTF-8 seq $betal $betas_stepsize $betah)
    do
        for j in $(LC_NUMERIC=en_US.UTF-8 seq $hl $hs_stepsize $hh)
        do
            ./main $i $j $num_iters $restart_num $ne "$outfile_stem"_b\_"$i" --max-reverse $max_reverse --PE-time $pe_time --PE-steps $pe_steps --X-mat-stem $X_mat_stem $reverse_count &
        done
    done
    wait
    echo "Launcher finished!"
fi
