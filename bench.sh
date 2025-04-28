VERSION=WORKER

# TMPDIR=build srun --partition=common --time=10 --gres=gpu:rtx2080ti -- \
TMPDIR=build srun --partition=common --time=10 --nodelist=asusgpu5 --gres=gpu:rtx2080ti -- \
    /usr/local/cuda/bin/ncu --set full --export profiler_output.ncu-rep --force-overwrite \
    ./acotsp tests-small/euc2d-pr1002.tsp profile.out ${VERSION} 1 1 1.1 0.9 42
