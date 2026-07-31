#pragma once

#include <random>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _status = (call);                                         \
        if (_status != cudaSuccess) {                                         \
            std::cerr << "CUDA ERROR: " << cudaGetErrorString(_status)        \
                      << " (code " << _status << ") "                         \
                      << "in " << __FILE__ << ':' << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)


void cuda_warmup();

void ColumnAddProjectionKernel_SRDW_1_NP_func(
    const float* d_flat_data,
    int* d_selected_examples,//selected examples indices
    float* d_col_add_projected,
    const int* d_offset,
    const int* d_flat_projection_col_idx,
    const float* d_flat_projection_weights,
    const int* d_node_row_off,
    int num_nodes,
    int num_proj,
    int num_rows,
    int selected_features_count);

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
                              bool verbose);

void ApplyProjectionColumnADD (const float* d_flat_data,
                                unsigned int* d_selected_examples,//selected examples indices
                                float* d_col_add_projected,
                                float** d_min_vals_out,
                                float** d_max_vals_out,
                                float** d_bin_widths_out,
                                std::vector<std::vector<std::vector<int>>>& projection_col_idx,
                                std::vector<std::vector<std::vector<float>>>& projection_weights,
                                const int num_rows,  //num_rows
                                const int num_proj, //num_proj
                                const unsigned int train_dataset,
                                double* elapsed_ms,
                                const int gpu_mode, //0: Exact, 1: Random, 2: Equal Width
                                const bool verbose,
                                std::vector<int>& node_start_idx,
                                std::vector<int>& node_count,
                                const bool separate,
                                const bool cub,
                                const bool fused,
                                int num_elems_per_thread
                              );

void RandomHistogram (const float* __restrict__ d_col_add_projected, //attributes
const int* __restrict__ selected_examples, //selected examples
const int* __restrict__ d_labels,
float* h_min_vals,
float* h_max_vals,
int** d_prefix_0,
int** d_prefix_1,
int** d_prefix_2,
float** d_candidate_splits,
const int num_rows, //selected_examples.size()
const int num_bins,
const int num_proj,
const int num_nodes,
std::vector<int>& node_start_idx, // *** NEW
std::vector<int>& node_count,
std::mt19937& random,
const bool verbose
);

void ThrustSortIndicesOnly(float* d_proj_values, 
                          unsigned int* d_row_ids,
                          unsigned int* d_selected_examples, 
                          int num_rows, 
                          int num_proj,
                          std::vector<int>& node_start_idx,
                          std::vector<int>& node_count,
                          bool verbose
                          );

void ExactSplit(
    unsigned int* d_sorted_indices,  // [num_proj * num_rows]
    const unsigned int* d_labels,          // [num_proj * num_rows]
    float* best_gain_out, // [num_proj], initial best gain values
    int* best_split_out, // [num_proj], initial best split values
    float* best_threshold_out,
    int* best_proj,
    const int num_rows,
    const int num_proj,
    float* d_col_add_projected,  // [num_proj * num_rows]
    double* elapsed_ms,
    bool verbose,
    const int comp_method,
    unsigned int* d_selected_examples,
    std::vector<int>& node_start_idx,
    std::vector<int>& node_count
    );

void ApplyProjectionColumnADDFused (const float* d_flat_data,
                                    unsigned int* d_selected_examples,//selected examples indices
                                    float* d_col_add_projected,
                                    float** d_min_vals_out,
                                    float** d_max_vals_out,
                                    float** d_bin_widths_out,
                                    std::vector<std::vector<std::vector<int>>>& projection_col_idx,
                                    std::vector<std::vector<std::vector<float>>>& projection_weights,
                                    const int num_selected_examples,  //num_rows
                                    const int num_proj, //num_proj
                                    const unsigned int num_total_rows,
                                    double* elapsed_apply_ms,
                                    const int gpu_mode, //0: Exact, 1: Random, 2: Equal Width
                                    const bool verbose,
                                    std::vector<int>& node_start_idx,
                                    std::vector<int>& node_count,
                                    const bool separate,
                                    const bool cub,
                                    const bool fused,
                                    const int num_bins,
                                    const int num_nodes,
                                    const unsigned int* __restrict__ d_labels,
                                    float*  h_min_vals,
                                    float*  h_max_vals,
                                    int**   d_prefix_0_out,
                                    int**   d_prefix_1_out,
                                    int**   d_prefix_2_out,
                                    float** d_candidate_splits_out,
                                    std::mt19937& random,
                                    const int col_gen_seed,
                                    const bool one_kernel,
                                    int num_elems_per_thread
                                    );
