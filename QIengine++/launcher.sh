if [ $# -lt 7 ]
then
    echo usage:
    echo ./launcher.sh "<lower beta>" "<higher beta>" "<beta stepsize>" "<num iterations>" "<restart steps>" "<stem outfile>" "<max reverse>"
else
    betal=$1
    betah=$2
    betas_stepsize=$3
    num_iters=$4
    restart_num=$5
    outfile_stem=$6
    max_reverse=$7
    for i in $(LC_NUMERIC=en_US.UTF-8 seq $betal $betas_stepsize $betah)
    do
        ./main $i 1.0 $num_iters $restart_num "$outfile_stem"_b\_"$i" --max-reverse $max_reverse &
    done
    wait
    echo "Launcher finished!"
fi
