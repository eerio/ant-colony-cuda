make
srun --partition=common --time 10 --gres=gpu:rtx2080ti -- ./acotsp tests/euc2d-pr1002.tsp tests/euc2d-pr1002_worker.out WORKER 20 1 2 0.5 42
