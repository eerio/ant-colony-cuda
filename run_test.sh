# ensure there is an argument
if [ $# -eq 0 ]; then
    echo "No arguments provided"
    exit 1
fi

# extract basename from the first argument
filename=$(basename -- "$1" ".${1##*.}")

make && \
srun --partition=common --time 10 --gres=gpu:1 -- ./acotsp "$1" out.txt WORKER 1 1 2 0.5 422 && \
head -n 1 out.txt && \
cat tsplib/solutions | grep "$filename" && \
echo ""