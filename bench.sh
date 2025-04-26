TMPDIR=build srun --partition=common --time=10 --gres=gpu:1 -- \
    /usr/local/cuda/bin/ncu --set full --export profiler_output --force-overwrite \
    ./acotsp tests/euc2d-d657.tsp tests/euc2d-d657_queen.out QUEEN 3 1 2 0.5 42
