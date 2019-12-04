if [ $# -lt 15 ]
then
    echo usage:
    echo ./launcher.sh "<lower beta>" "<higher beta>" "<beta stepsize>" "<lower h>" "<higher h>" "<h stepsize>"  "<num iterations>" "<restart steps>" "<state qbits>" "<ene qbits>" "<stem outfile>" "<max reverse>" "<PE time>" "<PE steps>" "<stem X matrix>" "<reverse_count>"
else
    betal=$1
    betah=$2
    betas_stepsize=$3
    hl=$4
    hh=$5
    hs_stepsize=$6
    num_iters=$7
    restart_num=$8
    ns=$9
    ne=${10}
    outfile_stem=${11}
    max_reverse=${12}
    pe_time=${13}
    pe_steps=${14}
    X_mat_stem=${15}
    reverse_count=${16}
    for i in $(LC_NUMERIC=en_US.UTF-8 seq $betal $betas_stepsize $betah)
    do
        for j in $(LC_NUMERIC=en_US.UTF-8 seq $hl $hs_stepsize $hh)
        do
            if [ -z "$X_mat_stem" ]
            then
                ./main $i $j $num_iters $restart_num $ns $ne "$outfile_stem"_b\_"$i" --max-reverse $max_reverse --PE-time $pe_time --PE-steps $pe_steps $reverse_count &
            else
                ./main $i $j $num_iters $restart_num $ns $ne "$outfile_stem"_b\_"$i" --max-reverse $max_reverse --PE-time $pe_time --PE-steps $pe_steps --X-mat-stem $X_mat_stem $reverse_count &
            fi
        done
    done
    wait
    echo "Launcher finished!"
fi
