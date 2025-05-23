
NX ?= 60
NY ?= 60
TAU ?= -0.01
EPS ?= 0.01
MAX_ITER ?= 10000
HEIGHT = 20
WIDTH = 30

default: solver

.PHONY: clean run show_image

run: solver
	./solver --nx $(NX) --ny $(NY) --tau $(TAU) --eps $(EPS) --max-iter $(MAX_ITER)

show_image: result.dat
	gnuplot -e "plot 'result.dat' binary \
		array=($(WIDTH), $(HEIGHT)) format='%lf' with image; pause -1"

solver: main.cu
	nvcc -O3 -arch=sm_70 -std=c++17 --compiler-options="-Wall" -o $@ $^ -lm -lboost_program_options

clean:
	rm -f solver result.dat