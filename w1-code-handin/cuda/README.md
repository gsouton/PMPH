# Task 3

### Compiling
You can build the code by using `make`.

### Use

Then the executable `./wa1-task3`, will run the task on cpu and gpu with some
default parameters

```c
#define SIZE 753411 //default size for the array
#define CPU_RUN 100 // number of time the task will be executed on CPU
#define GPU_RUN 100 // number of time the task will be executed on GPU
#define BLOCK_SIZE 256
#define MAX_BLOCK_SIZE 1024
#define EPSILON 0.0001 
```

Otherwise you can pass the array size and the block size use for the gpu 
as parameters:
- `./wa1-task3 <array_size> <gpu_block_size>` 

### Explanations

