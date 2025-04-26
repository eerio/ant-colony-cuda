# ensure there is an argument
if [ $# -eq 0 ]; then
    echo "No arguments provided"
    exit 1
fi

# extract basename without extension and directory
basename=$(basename -- "$1" ".${1##*.}")
# extract directory path
dir=$(dirname -- "$1")
# make output file path
output_file="${dir}/${basename}_worker.out"

make && \
srun --partition=common --time 10 --gres=gpu:1 -- ./acotsp "$1" "$output_file" WORKER 1 1 2 0.5 422 && \
head -n 1 "$output_file" && \
grep "$basename" tsplib/solutions