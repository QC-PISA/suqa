
betal=$1
betah=$2
betas_stepsize=$3
num_iters=$4
restart_num=$5
outfile_stem=$6
max_reverse=$7
for i in $(seq $betal $betas_stepsize $betah)
do
    ./main $i 1.0 $num_iters $restart_num "$outfile_stem"_b\_"$i" --max-reverse $max_reverse &
done
wait
echo "Launcher finished!"
