NVCC = /usr/local/cuda/bin/nvcc
# NVCCFLAGS = -O3 -Iinclude -DDEBUG # For Titan V (compute capability 7.0)
# NVCCFLAGS = -O3 -Iinclude -G -g -DMAX_SHMEM_SIZE=24576 # For Titan V (compute capability 7.0)

TSP_FILES=$(wildcard tests-small/*.tsp)

NVCCFLAGS = -O3 -Iinclude -arch=sm_75 # For Titan V (compute capability 7.0)
# SRUN_FLAGS = --partition=common --time 10 --nodelist=asusgpu6 --gres=gpu:rtx2080ti
SRUN_FLAGS = --partition=common --time 10 --nodelist=asusgpu5 --gres=gpu:rtx2080ti

# NVCCFLAGS = -O3 -Iinclude -G -g -arch=sm_70 # For Titan V (compute capability 7.0)
# SRUN_FLAGS = --partition=common --time 10 --nodelist=sylvester,steven,asusgpu3,asusgpu1 --gres=gpu:titanv
# SRUN_FLAGS = --partition=common --time 10 --nodelist=asusgpu3 --gres=gpu:titanv

# 100 iters, alpha 1, beta 1.1, rho 0.9, seed 42
TSP_FLAGS = 10 1 2 0.5 42

all: acotsp

# TMPDIR=build srun --partition=common --time 10 --gres=gpu:titanv -- $(NVCC) $(NVCCFLAGS) --device-c $< -o $@
%.o: %.cu tsp.cu
	TMPDIR=build srun $(SRUN_FLAGS) -- $(NVCC) $(NVCCFLAGS) --device-c $< -o $@

# TMPDIR=build srun --partition=common --time 10 --gres=gpu:titanv -- $(NVCC) $(NVCCFLAGS) $^ -o $@
acotsp: main.o worker.o queen.o baseline.o tsp.o
	TMPDIR=build srun $(SRUN_FLAGS) -- $(NVCC) $(NVCCFLAGS) $^ -o $@

clean:
	rm -rf *.o test balawender.zip acotsp out.txt

balawender.zip: *.cu Makefile include/*.h
	rm -f balawender.zip
	mkdir -p balawender
	cp -r *.cu include Makefile balawender/
	zip -r balawender.zip balawender
	rm -r balawender

pack: balawender.zip
	@echo "Packaged balawender.zip with source files and Makefile."

# test/balawender/acotsp: balawender.zip
# 	rm -rf test && \
# 	mkdir -p test && \
# 	cp balawender.zip test/ && \
# 	cd test && \
# 	unzip balawender.zip && \
# 	cd balawender && \
# 	cp -r ../../tsplib/ . && \
# 	make clean && \
# 	make && \
# 	TMPDIR=build srun $(SRUN_FLAGS) --  ./acotsp tsplib/a280.tsp out.txt BASELINE $(TSP_FLAGS); \
# 	head -n 1 out.txt; \
# 	cat tsplib/solutions | grep a280 ; \

test: test/balawender/acotsp

# run_ortools_parallel: $(TSP_FILES:.tsp=_ortools.out)
# tests/%_ortools.out: tests/%.tsp
# 	. .venv/bin/activate && python tsp_ortools.py $< > $@

# run_ortools_parallel_geo: $(GEO_TSP_FILES:.tsp=_ortools.out)
# tests/%_ortools.out: tests/%.tsp
# 	. .venv/bin/activate && python tsp_ortools.py $< > $@

run_baseline_parallel: acotsp $(TSP_FILES:.tsp=_baseline.out)
tests-small/%_baseline.out: tests/%.tsp
	./acotsp $< $@ BASELINE $(TSP_FLAGS)

run_worker_parallel: acotsp $(TSP_FILES:.tsp=_worker.out)
# run_worker_parallel: acotsp tests/euc2d-pr1002_worker.out tests/euc2d-d1291_worker.out tests/geo-gr96_worker.out
tests-small/%_worker.out: tests-small/%.tsp
	./acotsp $< $@ WORKER $(TSP_FLAGS)

# run_queen_parallel: acotsp tests/euc2d-pr1002_queen.out tests/euc2d-d657_queen.out tests/geo-gr96_queen.out
run_queen_parallel: acotsp $(TSP_FILES:.tsp=_queen.out)
tests-small/%_queen.out: tests-small/%.tsp
	./acotsp $< $@ QUEEN $(TSP_FLAGS)

TSPLIB_SOLUTIONS = tsplib/solutions

tests/%_tsplibsolution.out: tests/%.tsp $(TSPLIB_SOLUTIONS)
	@full="$*"; \
	name=$$(echo "$$full" | sed 's/^[^\-]*-//'); \
	solution_line=$$(grep "^$$name :" $(TSPLIB_SOLUTIONS)); \
	if [ -z "$$solution_line" ]; then \
		echo "Solution for $$name not found!" >&2; \
	fi; \
	cost=$$(echo $$solution_line | awk -F'[: ]+' '{print $$2}'); \
	echo "$$cost" > $@
all-tsplib-solutions: $(patsubst tests/%.tsp, tests/%_tsplibsolution.out, $(TSP_FILES))

rerun_worker:
	rm tests-small/*_worker.out; TMPDIR=build srun $(SRUN_FLAGS) -- make run_worker_parallel

rerun_queen:
	rm tests-small/*_queen.out; TMPDIR=build srun $(SRUN_FLAGS) -- make run_queen_parallel

check_worker:
	python3 check_results_worker.py

check_queen:
	python3 check_results_queen.py

.PHONY: all clean pack test ortools