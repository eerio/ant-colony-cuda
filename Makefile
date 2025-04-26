NVCC = /usr/local/cuda/bin/nvcc
# NVCCFLAGS = -O3 -Iinclude -DDEBUG # For Titan V (compute capability 7.0)
NVCCFLAGS = -O3 -Iinclude # For Titan V (compute capability 7.0)

TSP_FILES=$(wildcard tests/*.tsp)
GEO_TSP_FILES=$(wildcard tests/geo-*.tsp)

all: acotsp

%.o: %.cu tsp.cu
	# TMPDIR=build srun --partition=common --time 10 --gres=gpu:titanv -- $(NVCC) $(NVCCFLAGS) --device-c $< -o $@
	TMPDIR=build srun --partition=common --time 10 --gres=gpu:rtx2080ti -- $(NVCC) $(NVCCFLAGS) --device-c $< -o $@

acotsp: main.o worker.o queen.o baseline.o tsp.o
	# TMPDIR=build srun --partition=common --time 10 --gres=gpu:titanv -- $(NVCC) $(NVCCFLAGS) $^ -o $@
	TMPDIR=build srun --partition=common --time 10 --gres=gpu:rtx2080ti -- $(NVCC) $(NVCCFLAGS) $^ -o $@

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

test/balawender/acotsp: balawender.zip
	rm -rf test && \
	mkdir -p test && \
	cp balawender.zip test/ && \
	cd test && \
	unzip balawender.zip && \
	cd balawender && \
	cp -r ../../tsplib/ . && \
	make clean && \
	make && \
	srun --partition=common --time 10 --gres=gpu:1 -- ./acotsp tsplib/a280.tsp out.txt BASELINE 10 1 2 0.5 42; \
	head -n 1 out.txt; \
	cat tsplib/solutions | grep a280 ; \

test: test/balawender/acotsp

# run_ortools_parallel: $(TSP_FILES:.tsp=_ortools.out)
# tests/%_ortools.out: tests/%.tsp
# 	. .venv/bin/activate && python tsp_ortools.py $< > $@

# run_ortools_parallel_geo: $(GEO_TSP_FILES:.tsp=_ortools.out)
# tests/%_ortools.out: tests/%.tsp
# 	. .venv/bin/activate && python tsp_ortools.py $< > $@

run_baseline_parallel: acotsp $(TSP_FILES:.tsp=_baseline.out)
tests/%_baseline.out: tests/%.tsp
	./acotsp $< $@ BASELINE 5 10 2 0.5 42

run_worker_parallel: acotsp $(TSP_FILES:.tsp=_worker.out)
tests/%_worker.out: tests/%.tsp
	./acotsp $< $@ WORKER 10 1 2 0.5 425

run_queen_parallel: acotsp $(TSP_FILES:.tsp=_queen.out)
tests/%_queen.out: tests/%.tsp
	./acotsp $< $@ QUEEN 1000 1 2 0.5 425

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
	rm tests/*_worker.out; srun --partition=common --time 10 --gres=gpu -- make run_worker_parallel

rerun_queen:
	rm tests/*_queen.out; srun --partition=common --time 10 --gres=gpu -- make run_queen_parallel

check_results:
	python3 check_results_queen.py

.PHONY: all clean pack test ortools