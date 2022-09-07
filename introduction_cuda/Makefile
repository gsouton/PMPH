CC = nvcc
FLAGS = -O3

init: 
	mkdir bin

all: intro time


intro: intro.cu
	$(CC) $(FLAGS) $< -o $@ 

time: time.cu
	$(CC) $(FLAGS) $< -o $@

.PHONY: clean

clean: 
	rm -r bin

