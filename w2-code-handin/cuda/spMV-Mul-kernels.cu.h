#ifndef SP_MV_MUL_KERS
#define SP_MV_MUL_KERS

__global__ void
replicate0(int tot_size, char* flags_d) {
    const unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id < tot_size){
        flags_d[global_id] = 0;
    }
}


__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    const unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id < mat_rows){
        flags_d[mat_shp_sc_d[global_id]] = 1;
    }
}

__global__ void 
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    const unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id < tot_size){
        tmp_pairs[global_id] = mat_vals[global_id] * vct[mat_inds[global_id]];
    }
}

__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    const unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id < mat_rows){
        res_vct_d[global_id] = tmp_scan[mat_shp_sc_d[global_id]-1];
    }
}

#endif
