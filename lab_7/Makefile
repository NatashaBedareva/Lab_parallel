
NX ?= 100
NY ?= 100
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

solver: main.cpp
	pgc++ -acc=gpu -Minfo=all -std=c++17 -lm -lboost_program_options -lcublas -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/lib64/ -o solver main.cpp

clean:
	rm -f solver result.dat