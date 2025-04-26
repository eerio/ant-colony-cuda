NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -O3 -Iinclude -arch=sm_70 -DDEBUG # For Titan V (compute capability 7.0)

TSP_FILES=$(wildcard tests/*.tsp)

all: acotsp

%.o: %.cu
	# TMPDIR=build srun --partition=common --time 10 --gres=gpu:titanv -- $(NVCC) $(NVCCFLAGS) -c $< -o $@
	TMPDIR=build srun --partition=common --time 10 --gres=gpu -- $(NVCC) $(NVCCFLAGS) -c $< -o $@

acotsp: main.o worker.o queen.o baseline.o
	# TMPDIR=build srun --partition=common --time 10 --gres=gpu:titanv -- $(NVCC) $(NVCCFLAGS) $^ -o $@
	TMPDIR=build srun --partition=common --time 10 --gres=gpu -- $(NVCC) $(NVCCFLAGS) $^ -o $@

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

run_ortools_parallel: $(TSP_FILES:.tsp=_ortools.out)
tests/%_ortools.out: tests/%.tsp
	. .venv/bin/activate && python tsp_ortools.py $< > $@

run_baseline_parallel: acotsp $(TSP_FILES:.tsp=_baseline.out)
tests/%_baseline.out: tests/%.tsp
	./acotsp $< $@ BASELINE 5 10 2 0.5 42

.PHONY: all clean pack test ortools