NVCC = /usr/local/cuda/bin/nvcc
# NVCCFLAGS = -O3 -arch=sm_75  # For Titan V (compute capability 7.5)
NVCCFLAGS = -O3 # For Titan V (compute capability 7.5)

all: acotsp

%.o: %.cu
	TMPDIR=build srun --partition=common --time 10 --gres=gpu:1 -- $(NVCC) $(NVCCFLAGS) -c $< -o $@

acotsp: main.o
	TMPDIR=build srun --partition=common --time 10 --gres=gpu:1 -- $(NVCC) $(NVCCFLAGS) $^ -o $@

clean:
	rm -rf *.o test balawender.zip acotsp out.txt

balawender.zip: *.cu Makefile
	rm -f balawender.zip
	mkdir -p balawender
	cp *.cu Makefile balawender/
	zip -r balawender.zip balawender
	rm -r balawender

pack: balawender.zip
	@echo "Packaged balawender.zip with source files and Makefile."

test: pack
	mkdir -p test
	cp balawender.zip test/
	cd test && \
		unzip balawender.zip && \
		cd balawender && \
		cp ../../tsplib/a280.tsp . && \
		make clean && \
		make && \
		srun --partition=common --time 10 --gres=gpu:1 -- ./acotsp a280.tsp out.txt WORKER 1 1 2 0.5 42; \
		cd ../.. && \
		rm -rf test


.PHONY: all clean pack test