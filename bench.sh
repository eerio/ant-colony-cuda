VERSION=WORKER

# TMPDIR=build srun --partition=common --time=10 --gres=gpu:rtx2080ti -- \
TMPDIR=build srun --partition=common --time=10 --nodelist=asusgpu6 --gres=gpu:rtx2080ti -- \
    /usr/local/cuda/bin/ncu --set full --export profiler_output.ncu-rep --force-overwrite \
    ./acotsp tests/euc2d-d657.tsp tests/euc2d-d657_${VERSION,,}.out ${VERSION} 1 1 2 0.5 42
