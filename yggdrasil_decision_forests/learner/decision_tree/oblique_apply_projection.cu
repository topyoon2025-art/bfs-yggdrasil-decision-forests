#include <cuda_runtime.h>
#include <iostream>
#include <chrono>



__global__  void warmup_kernel() {
    // This kernel does nothing but can be used to warm up the GPU
    // to avoid including initialization time in our benchmarks.
}

void cuda_warmup() {
    warmup_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}


template<int BLOCK>
__global__ void ColumnAddProjectionKernel_SRDW_1(
        const float*        __restrict__ dataset,
        const int* __restrict__ d_selected_examples,
        float*              __restrict__ projected,
        const int*          __restrict__ col_offset,
        const int*          __restrict__ flat_col_data,
        const float*        __restrict__ flat_weights,
        const int*          __restrict__ node_row_off,
        int                 num_nodes,
        int                 num_proj,
        int                 num_rows,
        int                 selected_features_count) // Added num_nodes parameter   
{   
           
    const int node_id = blockIdx.y;
    const int proj_id = blockIdx.z;

    const int global_row = blockIdx.x * blockDim.x + threadIdx.x; // global row index for all nodes
    const int row_stride = gridDim.x * blockDim.x;

    // const int seg_id  = proj_id * num_nodes + node_id;
    const int seg_id  = node_id * num_proj + proj_id; // Changed to node-major order for col_offset

    const int row_start = node_row_off[node_id];
    const int row_end   = node_row_off[node_id + 1];
    const int rows_node = row_end - row_start;
    // Or just pass in the node counts per node? then we read once per thread instead of twice. 

    const int begin = col_offset[seg_id]; // can get rid of this if offset is uniform across all nodes, but for now we keep it general
    const int end   = col_offset[seg_id + 1];

    const int base = row_start * num_proj + proj_id * rows_node;
    for (int r = global_row; r < rows_node; r += row_stride)
    {
        const int ex_idx = d_selected_examples[row_start + r]; //dense read
        if (ex_idx < 0 || ex_idx >= num_rows) continue;
            float sum = 0.0f;

        for (int idx = begin; idx < end; ++idx)
        {
            float w   = flat_weights[idx];
            int   col = flat_col_data[idx];

            // Calculate the index into the dataset array
            const std::size_t dataset_idx = static_cast<std::size_t>(col) * num_rows + ex_idx;
            const float x = dataset[dataset_idx];
            sum += w * x;
            
        }
        // output layout: node-major, then projection, then row
        projected[ base + r ] = sum;
    }
}

void ColumnAddProjectionKernel_SRDW_1_NP_func(
    const float* d_flat_data,
    int* d_selected_examples,
    float* d_col_add_projected,
    const int* d_offset,
    const int* d_flat_projection_col_idx,
    const float* d_flat_projection_weights,
    const int* d_node_row_off,
    int num_nodes,
    int num_proj,
    int num_rows,
    int selected_features_count,
    int max_rows_per_node) {

    constexpr int BLOCK = 256;
    int blocks_per_node = (max_rows_per_node + BLOCK - 1) / BLOCK;
    dim3 grid(blocks_per_node, num_nodes, num_proj);

    ColumnAddProjectionKernel_SRDW_1<BLOCK><<<grid, BLOCK>>>(
        d_flat_data, d_selected_examples, d_col_add_projected,
        d_offset, d_flat_projection_col_idx, d_flat_projection_weights,
        d_node_row_off, num_nodes, num_proj, num_rows,
        selected_features_count);
    cudaDeviceSynchronize();
}

template<int BLOCK>
__global__ void ColumnAddProjectionKernel_SRDW_2(
        const float*        __restrict__ dataset,
        const int* __restrict__ d_selected_examples,
        float*              __restrict__ projected,
        const int*          __restrict__ col_offset,
        const int*          __restrict__ flat_col_data,
        const float*        __restrict__ flat_weights,
        const int*          __restrict__ node_row_off,
        int                 num_nodes,
        int                 num_proj,
        int                 num_total_rows,
        int                 selected_features_count) // Added num_nodes parameter   
{   
           
    const int node_id = blockIdx.z;
    const int proj_id = blockIdx.y;

    const int global_row = blockIdx.x * blockDim.x + threadIdx.x; // global row index for all nodes
    const int row_stride = gridDim.x * blockDim.x;

    // const int seg_id  = proj_id * num_nodes + node_id; 
    const int seg_id  = node_id * num_proj + proj_id; // Changed to node-major order for col_offset
 
    const int row_start = node_row_off[node_id];
    const int row_end   = node_row_off[node_id + 1];
    const int rows_node = row_end - row_start;
    // Or just pass in the node counts per node? then we read once per thread instead of twice. 

    const int begin = col_offset[seg_id]; // can get rid of this if offset is uniform across all nodes, but for now we keep it general
    const int end   = col_offset[seg_id + 1]; 

    const int base = row_start * num_proj + proj_id * rows_node;
    for (int r = global_row; r < rows_node; r += row_stride)
    {
        const int ex_idx = d_selected_examples[row_start + r]; //dense read
        if (ex_idx < 0 || ex_idx >= num_total_rows) continue;
        float sum = 0.0f;

        for (int idx = begin; idx < end; ++idx){
            float w   = flat_weights[idx];
            int   col = flat_col_data[idx];
            const std::size_t dataset_idx = static_cast<std::size_t>(col) * num_total_rows + ex_idx; //sparse read
            const float x = dataset[dataset_idx];
            sum += w * x; 
        }
        projected[ base + r ] = sum; //dense write
    }
}


template<int BLOCK>
__global__ void ColumnAddProjectionKernel_DRSW_1(
        const float*        __restrict__ dataset,
        float*              __restrict__ projected,
        const int*          __restrict__ node_ids,
        const int*          __restrict__ node_offsets,
        const int*          __restrict__ col_offset,
        const int*          __restrict__ flat_col_data,
        const float*        __restrict__ flat_weights,
        const int*          __restrict__ node_row_off,
        int                 num_proj,
        int                 num_total_rows,
        int                 num_nodes,
        int*                node_counts,
        int*                node_row_start_by_row,
        int                 selected_features_count) // Added num_nodes parameter
{
    const int proj_id = blockIdx.y;
    const int row_stride = gridDim.x * blockDim.x;
    const int global_row = blockIdx.x * blockDim.x + threadIdx.x; // global row index for all nodes
    // if (global_row >= num_total_rows) return; // out of bounds check
 
    for (int r = global_row; r < num_total_rows; r += row_stride)
    {
        const int node_id = node_ids[r]; // node_id for this block
        const int row_start = node_row_off[node_id]; // Prefix sum of rows for each node, gives us the starting index in the output for this node
        const int row_end   = node_row_off[node_id + 1];  // node_row_off size is num_nodes + 1, so this is safe
        const int rows_node = row_end - row_start; // number of rows for this node
        // Or just pass in the node counts per node? then we read once per thread instead of twice. 

        const int node_offset = node_offsets[r]; // offset in the output for this thread
        
        // const int seg_id = proj_id * num_nodes + node_id;
        const int seg_id = node_id * num_proj + proj_id; // Changed to node-major order for col_offset

        const int begin = col_offset[seg_id]; // can get rid of this if offset is uniform across all nodes, but for now we keep it general
        const int end   = col_offset[seg_id + 1]; 

        float sum = 0.0f;
        for (int idx = begin; idx < end; ++idx) {
            int   col   = flat_col_data[idx]; // flat_col_data size: num_nodes * num_proj * selected_features_count
            float w     = flat_weights[idx];
            const std::size_t dataset_idx = static_cast<std::size_t>(col) * num_total_rows + r;
            const float x = dataset[dataset_idx];
            sum += w * x;
        }
        // output layout: node-major, then projection, then row
        const int base = row_start * num_proj + proj_id * rows_node;
        projected[base + node_offset] = sum;
    }
}


void ApplyProjectionBaseline (const float* d_flat_data,
                              int* d_selected_examples,//selected examples indices
                              float* d_col_add_projected,
                              int* d_node_ids,
                              int* d_node_offsets,
                              int* d_node_counts,
                              int* d_node_row_start_by_row,
                              const int num_proj,
                              const int num_total_rows,
                              const int num_segments,
                              const int num_nodes,
                              const int num_elems_per_thread,
                              std::vector<int>& node_start_idx,
                              std::vector<int>& node_count,
                              int selected_features_count,
                              bool alternate,
                              float* d_min_vals,
                              float* d_max_vals,
                              int max_rows_per_node,
                              int* node_row_off,
                              int* d_node_row_off,
                              int* d_offset,
                              int* d_flat_projection_col_idx,
                              float* d_flat_projection_weights,
                              int* d_flat_projection_col_idx_shared,
                              float* d_flat_projection_weights_shared,
                              bool verbose)//num_nodes
{
    warmup_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    
    constexpr int BLOCK = 256; 
    
    int rows_per_block = BLOCK * num_elems_per_thread;
    int blocks_per_all_nodes_per_proj = (num_total_rows + rows_per_block - 1) / rows_per_block;
    int blocks_per_node = (max_rows_per_node + rows_per_block - 1) / rows_per_block;

    if (alternate) {
        // Launch the alternative kernel that uses node_ids and node_offsets
        dim3 gridAP_DRSW_1(blocks_per_all_nodes_per_proj, num_proj);
        auto start_kernel_DRSW_1 = std::chrono::steady_clock::now();
        ColumnAddProjectionKernel_DRSW_1<BLOCK><<<gridAP_DRSW_1, BLOCK>>>(
            d_flat_data,                     // your original device dataset
            d_col_add_projected,             // output buffer (device)
            d_node_ids,
            d_node_offsets,
            d_offset,                        // column offsets
            d_flat_projection_col_idx,
            d_flat_projection_weights,
            d_node_row_off,                  // per-node row prefix
            num_proj,
            num_total_rows,
            num_nodes,
            d_node_counts,
            d_node_row_start_by_row,
            selected_features_count); // Pass num_nodes to the kernel
        cudaDeviceSynchronize();
        auto end_kernel_DRSW_1 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> kernel_DRSW_1_duration = end_kernel_DRSW_1 - start_kernel_DRSW_1;
        std::cout << "Time taken for ColumnAddProjectionKernel_DRSW_1: " << kernel_DRSW_1_duration.count() << " ms" << std::endl;

    } 

    else { 

        dim3 gridAP_SRDW_1(blocks_per_node, num_nodes, num_proj); // 3D grid with node_id as the z-dimension
        auto start_kernel_SRDW_1 = std::chrono::steady_clock::now();
        ColumnAddProjectionKernel_SRDW_1<BLOCK><<<gridAP_SRDW_1, BLOCK>>>(
            d_flat_data,                     // your original device dataset
            d_selected_examples,             // flat rows (all nodes)
            d_col_add_projected,             // output buffer (device)
            d_offset,                        // column offsets
            d_flat_projection_col_idx,
            d_flat_projection_weights,
            d_node_row_off,                  // per-node row prefix
            num_nodes,
            num_proj,
            num_total_rows,
            selected_features_count); 
        cudaDeviceSynchronize();
        auto end_kernel_SRDW_1 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> kernel_SRDW_1_duration = end_kernel_SRDW_1 - start_kernel_SRDW_1;
        std::cout << "Time taken for ColumnAddProjectionKernel_SRDW_1: " << kernel_SRDW_1_duration.count() << " ms" << std::endl;


        dim3 gridAP_SRDW_2(blocks_per_node, num_proj, num_nodes); // 3D grid with node_id as the z-dimension
        auto start_kernel_SRDW_2 = std::chrono::steady_clock::now();
        ColumnAddProjectionKernel_SRDW_2<BLOCK><<<gridAP_SRDW_2, BLOCK>>>(
            d_flat_data,                     // your original device dataset
            d_selected_examples,             // flat rows (all nodes)
            d_col_add_projected,             // output buffer (device)
            d_offset,                        // column offsets
            d_flat_projection_col_idx,
            d_flat_projection_weights,
            d_node_row_off,                  // per-node row prefix
            num_nodes,
            num_proj,
            num_total_rows,
            selected_features_count); 
        cudaDeviceSynchronize();
        auto end_kernel_SRDW_2 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> kernel_SRDW_2_duration = end_kernel_SRDW_2 - start_kernel_SRDW_2;
        std::cout << "Time taken for ColumnAddProjectionKernel_SRDW_2: " << kernel_SRDW_2_duration.count() << " ms" << std::endl;

    }
}

