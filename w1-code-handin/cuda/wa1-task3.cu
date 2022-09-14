#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>

//----------------------------------------------------------------------------------------
//---------------------------------- CONSTANTS -------------------------------------------
//----------------------------------------------------------------------------------------
#define SIZE 753411 //default size for the array
#define CPU_RUN 100 // number of time the task will be executed on CPU
#define GPU_RUN 100 // number of time the task will be executed on GPU
#define BLOCK_SIZE 256
#define MAX_BLOCK_SIZE 1024
#define EPSILON 0.0001 

void usage(){
    fprintf(stderr, "Use: ./wa1-taks3.cu <array_size> <block_size>\n");
    exit(-1);
}

//----------------------------------------------------------------------------------------
//----------------------------------- STRUCTS --------------------------------------------
//----------------------------------------------------------------------------------------
typedef struct env {
    unsigned int array_size; 
    unsigned long mem_size; 
    unsigned int cpu_run; 
    unsigned int gpu_run; 
    unsigned int block_size; 
}env_t;

void log_env(env_t *env){
    printf("{\n - array_size: %d\n - mem_size: %d\n - cpu_run: %d\n - gpu_run: %d\n - block_size: %d\n}\n", 
            env->array_size, 
            env->mem_size,
            env->cpu_run,
            env->gpu_run,
            env->block_size);
}


void init_env(env_t *env, int argc, char **argv){
    int array_size = SIZE;
    int block_size = BLOCK_SIZE;
    if(argc == 3){
        if((array_size = atoi(argv[1])) <= 0){
            fprintf(stderr, "array size could not be parsed\n");
            usage();
        } else if((block_size = atoi(argv[2])) <= 0 || block_size > MAX_BLOCK_SIZE){
            fprintf(stderr, "block size could not be parsed, Note: block_size cannot be over 1024\n");
            usage();
        }
    }
    env->array_size = array_size;
    env->mem_size = sizeof(float)*array_size;
    env->cpu_run = CPU_RUN;
    env->gpu_run = GPU_RUN;
    env->block_size = block_size;
    log_env(env);
}

//----------------------------------------------------------------------------------------
//---------------------------- FUNCTION DECLARATIONS -------------------------------------
//----------------------------------------------------------------------------------------
//debug functions
void log_array(float* array, unsigned int size);
void log_time(unsigned long int elapsed);

// util functions
float* new_array(unsigned int size);
int timeval_subtract(struct timeval* result, struct timeval* t2,struct timeval* t1);
bool check_computation(float *cpu_array, float *gpu_array, unsigned int size, float epsilon);

//task related functions
unsigned long int execute_task_on_cpu(float *input, float* output, env_t* env);
unsigned long int execute_task_on_gpu(float* input, float* output, env_t* env);


//----------------------------------------------------------------------------------------
//--------------------------------------- KERNELS ---------------------------------------
//----------------------------------------------------------------------------------------
/**
 * Kernel that squares each element in an array
 * In haskell notation: 
 *  - map (\v -> v*v) array
 */
__global__ void squareKernel(float *d_in, float *d_out) {
    const unsigned int lid = threadIdx.x; // local id inside a block
    const unsigned int gid = blockIdx.x * blockDim.x + lid; // global id
    d_out[gid] = d_in[gid] * d_in[gid];                     // do computation
}

/**
 * Kernel that squares each element in an array
 * In haskell notation: 
 *  - map (\v -> (x/(x-2.3))^3 ) array
 */
__global__ void cubeKernel(float *d_in, float *d_out, unsigned int size ) {
    const unsigned int local_id = threadIdx.x; // local id inside a block
    const unsigned int global_id = blockIdx.x * blockDim.x + local_id; // global id
    if(global_id < size){
        float x = d_in[global_id]/(d_in[global_id]-2.3);
        d_out[global_id] = x*x*x;
    }
}



//----------------------------------------------------------------------------------------
//-------------------------------------- MAIN --------------------------------------------
//----------------------------------------------------------------------------------------

int main(int argc, char **argv) {
    //intialize environment
    env_t env;
    init_env(&env, argc, argv);
    
    //Initialize host memory
    float *host_in = new_array(env.array_size);
    for(unsigned int i = 0; i < env.array_size; i++){
        host_in[i] = i;
    }

    //----------- CPU EXECUTION ---------------
    printf("---------- Executing task on CPU -----------\n");
    float *cpu_array = new_array(env.array_size);
    unsigned long int cpu_time = execute_task_on_cpu(host_in, cpu_array, &env);
    log_time(cpu_time);
    printf("--------------------------------------------\n\n");

    //----------- GPU EXECUTION ---------------
    printf("---------- Executing task on GPU -----------\n");
    float *host_out = new_array(env.array_size);
    unsigned long int gpu_time = execute_task_on_gpu(host_in, host_out, &env);
    log_time(gpu_time);
    printf("--------------------------------------------\n\n");

    printf("----------- Checking result ----------------\n");
    if (!check_computation(cpu_array, host_out, SIZE, 0.0001)){
        printf("INVALID!\n");
    }else{
        printf("VALID!\n");
    }

    //Free memory
    free(host_in);
    free(host_out);
    free(cpu_array);
}

float* new_array(unsigned int size){
    float *array = (float *)calloc(sizeof(float), size);
    if(!array){
        fprintf(stderr, "Could not allocate array of size: %d", size); 
        exit(-1);
    }
    return array;
}

/**
 * Execute a task (3) on the cpu and return the elapsed time
 * The task is to map a the function (x/(x-2.3))^3 to each element of the array
 * 
 * @input: initial array
 * @output: array containing the result  
 */
unsigned long int execute_task_on_cpu(float *input, float *output, env_t *env){
    unsigned long int elapsed; struct timeval t_start, t_end, t_diff;

    gettimeofday(&t_start, NULL);
    for(unsigned int it = 0; it < env->cpu_run; ++it){
        for(unsigned int i = 0; i < env->array_size; ++i){
            float x = (input[i]/(input[i]-2.3));
            output[i] = x*x*x;
        }
    }
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / env->cpu_run;
    return elapsed;
}

/**
 * Execute a task (3) on the gpu and return the elapsed time
 * The task is to map a the function (x/(x-2.3))^3 to each element of the array
 * 
 * @input: initial array
 * @output: array containing the result  
 */
unsigned long int execute_task_on_gpu(float* host_in, float* host_out, env_t *env){
    unsigned int block_size = env->block_size;
    unsigned int gpu_run = env->gpu_run;
    assert(block_size <= MAX_BLOCK_SIZE);

    unsigned long int elapsed; struct timeval t_start, t_end, t_diff;

    //Compute number of block needed
    unsigned int num_blocks = (env->array_size + (block_size - 1)) / block_size;
    printf("{Block_size: %d, num_blocks: %d}\n", block_size, num_blocks);

    float *device_in;
    float *device_out;

    //// allocate device memory
    cudaMalloc((void **)&device_in, env->mem_size);
    cudaMalloc((void **)&device_out, env->mem_size);

    //// copy host memory to device
    cudaMemcpy(device_in, host_in, env->mem_size, cudaMemcpyHostToDevice);

    //// execute the kernel
    gettimeofday(&t_start, NULL);
    for(unsigned int i = 0; i < gpu_run; ++i){
        cubeKernel<<<num_blocks, block_size>>>(device_in, device_out, env->array_size);
    }cudaThreadSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / gpu_run;

    //// copy result from device to host
    cudaMemcpy(host_out, device_out, env->mem_size, cudaMemcpyDeviceToHost);

    //// clean-up memory
    cudaFree(device_in);
    cudaFree(device_out);
    return elapsed;
}


/**
 * Function use for comparing the result computed on the cpu and gpu*
 *
 * @cpu_array: array computed on the cpu
 * @gpu_array: array computed on the gpu
 * @size: size of @cpu_array and @gpu_array  
 * @epsilon: 
 */
bool check_computation(float *cpu_array, float *gpu_array, unsigned int size, float epsilon){
    for(unsigned int i = 0; i < size; ++i){
        if(fabs(cpu_array[i] - gpu_array[i]) >= epsilon){
            return false;
        }
    }
    return true;
}


void log_array(float *array, unsigned int size){
    assert(array != NULL);
    printf("array: %p\n", array);
    for(unsigned int i = 0; i < size; ++i){
        printf("%lf\n", array[i]);
    }
    printf("--------------------\n");
}

void log_time(unsigned long int elapsed){
    printf("- Took %d microseconds (%.2fms)\n", elapsed, elapsed/1000.0);
}


/**
 * Compute difference between timevals 
 */
int timeval_subtract(struct timeval* result, struct timeval* t2,struct timeval* t1) {
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) -
    (t1->tv_usec + resolution * t1->tv_sec) ;
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}
