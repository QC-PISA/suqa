if [ $# -lt 9 ]
then
    echo usage:
    echo ./launcher.sh "<lower beta>" "<higher beta>" "<beta stepsize>" "<num iterations>" "<restart steps>" "<stem outfile>" "<max reverse>" "<PE time>" "<PE steps>"
else
    betal=$1
    betah=$2
    betas_stepsize=$3
    num_iters=$4
    restart_num=$5
    outfile_stem=$6
    max_reverse=$7
    pe_time=$8
    pe_steps=$9
    for i in $(LC_NUMERIC=en_US.UTF-8 seq $betal $betas_stepsize $betah)
    do
        ./main $i $num_iters $restart_num "$outfile_stem"_b\_"$i" --max-reverse $max_reverse --PE-time $pe_time --PE-steps $pe_steps &
    done
    wait
    echo "Launcher finished!"
fi
