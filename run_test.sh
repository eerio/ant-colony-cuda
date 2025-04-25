make; \
srun --partition=common --time 10 --gres=gpu:1 -- ./acotsp tsplib/$1.tsp out.txt WORKER 1 1 2 0.5 42 && \
head -n 1 out.txt && \
cat tsplib/solutions | grep $1 && \
echo ""