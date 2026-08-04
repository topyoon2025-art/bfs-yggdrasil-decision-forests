#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <vector>

#include "yggdrasil_decision_forests/learner/decision_tree/maximize_groupings.h"

namespace Utils {

    struct ExperimentParams
    {
        int   selected_features_count;
        int   num_elems_per_thread;
        int   num_iters;
        int   num_nodes;
        int   num_proj;
        bool  alternate;
        bool  verbose;
        int   reorder_strategy = 0;
    };

    template <typename DataT, typename LabelT>
    struct CSVData {
        std::vector<DataT> flattened;
        std::vector<LabelT> labels;
        int num_rows;
        int num_cols;
    };

    template <typename DataT, typename LabelT>
    CSVData<DataT, LabelT> flattenCSVColumnMajorWithLabels(const std::string& filename) {
        static_assert(std::is_arithmetic<DataT>::value, "CSVData type must be numeric");

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return {{}, {}, 0, 0};
        }

        std::vector<std::vector<DataT>> rows;
        std::string line;

        // Skip header
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<DataT> row;

            while (std::getline(ss, cell, ',')) {
                try {
                    if constexpr (std::is_same<DataT, int>::value) {
                        row.push_back(std::stoi(cell));
                    } else if constexpr (std::is_same<DataT, float>::value) {
                        row.push_back(std::stof(cell));
                    } else if constexpr (std::is_same<DataT, double>::value) {
                        row.push_back(std::stod(cell));
                    } else {
                        row.push_back(static_cast<DataT>(std::stod(cell))); // fallback
                    }
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid value: " << cell << std::endl;
                    row.push_back(static_cast<DataT>(0)); // fallback
                }
            }

            rows.push_back(row);
        }

        file.close();

        if (rows.empty()) {
            std::cerr << "CSV is empty.\n";
            return {{}, {}, 0, 0};
        }

        int num_rows = static_cast<int>(rows.size());
        int num_cols = static_cast<int>(rows[0].size());
        size_t total_features = static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols - 1);
        std::vector<DataT> flattened;
        flattened.reserve(total_features);

        std::vector<LabelT> labels;
        labels.reserve(num_rows);

        for (int col = 0; col < num_cols - 1; ++col) {
            for (int row = 0; row < num_rows; ++row) {
                flattened.push_back(rows[row][col]);
            }
        }


        for (int row = 0; row < num_rows; ++row) {
            labels.push_back(rows[row][num_cols - 1]); // last column
        }

        return {flattened, labels, num_rows, num_cols - 1};
    }

    struct NodeSplit {
        std::vector<std::vector<int>> splits;
        std::vector<int> selected_examples;
        std::vector<int> node_start_idx;
        std::vector<int> node_counts;
        std::vector<int> node_ids;
        std::vector<int> node_offsets;
        std::vector<int> node_row_off;
        std::vector<int> node_row_start_by_row;
        std::vector<int> node_counts_by_row;
        int max_rows_per_node;
    };

    inline NodeSplit splitExamples(int num_rows, int num_nodes,
                                   unsigned seed, bool sort_chunks) {
        std::vector<int> indices(num_rows);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<int> starts(num_nodes);
        std::vector<int> counts(num_nodes);
        int base = num_rows / num_nodes;
        int rem  = num_rows % num_nodes;
        int pos  = 0;
        for (int i = 0; i < num_nodes; ++i) {
            counts[i] = base + (i < rem ? 1 : 0);
            starts[i] = pos;
            pos += counts[i];
        }

        std::vector<std::vector<int>> splits(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            auto lo = indices.begin() + starts[i];
            auto hi = lo + counts[i];
            if (sort_chunks) std::sort(lo, hi);
            splits[i].assign(lo, hi);
        }

        std::vector<int> nids(num_rows, -1);
        std::vector<int> noffs(num_rows, -1);
        for (int i = 0; i < num_nodes; ++i) {
            for (int j = 0; j < counts[i]; ++j) {
                int row = splits[i][j];
                nids[row] = i;
                noffs[row] = j;
            }
        }

        std::vector<int> row_off(num_nodes + 1);
        row_off[0] = starts[0];
        for (int i = 0; i < num_nodes; ++i)
            row_off[i + 1] = row_off[i] + counts[i];

        std::vector<int> row_start_by_row(num_rows, -1);
        for (int i = 0; i < num_nodes; ++i) {
            for (int j = 0; j < counts[i]; ++j)
                row_start_by_row[splits[i][j]] = row_off[i];
        }

        int max_rpn = 0;
        for (int i = 0; i < num_nodes; ++i)
            max_rpn = std::max(max_rpn, row_off[i + 1] - row_off[i]);

        std::vector<int> counts_by_row(num_rows, 0);
        for (int r = 0; r < num_rows; ++r) {
            int nid = nids[r];
            if (nid >= 0 && nid < num_nodes)
                counts_by_row[r] = row_off[nid + 1] - row_off[nid];
        }

        return {std::move(splits), std::move(indices), std::move(starts),
                std::move(counts), std::move(nids), std::move(noffs),
                std::move(row_off), std::move(row_start_by_row),
                std::move(counts_by_row), max_rpn};
    }

    struct RandomConfig {
        int count;
        int minValue;
        int maxValue;
        bool unique;
        uint64_t seed;
    };

    inline std::vector<int> generateRandom(const RandomConfig& cfg) {
        std::mt19937 rng(static_cast<unsigned>(cfg.seed));
        std::vector<int> result;
        result.reserve(cfg.count);

        if (cfg.unique) {
            std::vector<int> pool(cfg.maxValue - cfg.minValue + 1);
            std::iota(pool.begin(), pool.end(), cfg.minValue);
            std::shuffle(pool.begin(), pool.end(), rng);
            result.assign(pool.begin(), pool.begin() + cfg.count);
        } else {
            std::uniform_int_distribution<int> dist(cfg.minValue, cfg.maxValue);
            for (int i = 0; i < cfg.count; ++i)
                result.push_back(dist(rng));
        }
        return result;
    }

    struct Projections {
        std::vector<std::vector<std::vector<int>>>   col_idx;
        std::vector<std::vector<std::vector<float>>> weights;
    };

    inline Projections generateProjections(int num_proj, int num_nodes,
                                           int num_cols, int selected_features_count,
                                           uint64_t base_seed) {
        RandomConfig cfg;
        cfg.minValue = 0;
        cfg.maxValue = num_cols - 1;
        cfg.unique   = true;
        cfg.count    = selected_features_count;

        std::vector<std::vector<std::vector<int>>>   col_idx(num_nodes);
        std::vector<std::vector<std::vector<float>>> wts(num_nodes);

        for (int n = 0; n < num_nodes; ++n) {
            col_idx[n].resize(num_proj);
            wts[n].resize(num_proj);

            std::set<std::vector<int>> seen_sets;

            std::mt19937 weight_rng(static_cast<unsigned>(
                base_seed + static_cast<uint64_t>(n) * 9176ULL + 99991ULL));
            std::uniform_int_distribution<int> sign_dist(0, 1);

            for (int p = 0; p < num_proj; ++p) {
                uint64_t seed = base_seed
                    + static_cast<uint64_t>(p) * 1000003ULL
                    + static_cast<uint64_t>(n) * 9176ULL;

                std::vector<int> cols;
                while (true) {
                    cfg.seed = seed;
                    cols = generateRandom(cfg);
                    auto sorted = cols;
                    std::sort(sorted.begin(), sorted.end());
                    if (seen_sets.insert(sorted).second)
                        break;
                    ++seed;
                }
                col_idx[n][p] = std::move(cols);

                wts[n][p].resize(selected_features_count);
                for (int k = 0; k < selected_features_count; ++k)
                    wts[n][p][k] = sign_dist(weight_rng) == 0 ? -1.0f : 1.0f;
            }
        }

        return {std::move(col_idx), std::move(wts)};
    }

    struct FlatProjections {
        std::vector<int>   col_idx;
        std::vector<float> weights;
        std::vector<int>   offsets;
    };

    inline FlatProjections flattenProjections(const Projections& proj,
                                              int num_proj, int num_nodes) {
        int num_seg = num_nodes * num_proj;
        std::vector<int> col_per_proj(num_seg);
        int k = 0;
        for (int n = 0; n < num_nodes; ++n)
            for (int p = 0; p < num_proj; ++p, ++k)
                col_per_proj[k] = proj.col_idx[n][p].size();

        std::vector<int> offsets(num_seg + 1);
        offsets[0] = 0;
        if (num_seg > 0)
            std::inclusive_scan(col_per_proj.begin(), col_per_proj.end(),
                                offsets.begin() + 1);
        int total = offsets.back();

        std::vector<int>   flat_col(total);
        std::vector<float> flat_wts(total);
        k = 0;
        for (int n = 0; n < num_nodes; ++n) {
            for (int p = 0; p < num_proj; ++p, ++k) {
                std::memcpy(flat_col.data() + offsets[k],
                            proj.col_idx[n][p].data(),
                            proj.col_idx[n][p].size() * sizeof(int));
                std::memcpy(flat_wts.data() + offsets[k],
                            proj.weights[n][p].data(),
                            proj.weights[n][p].size() * sizeof(float));
            }
        }
        return {std::move(flat_col), std::move(flat_wts), std::move(offsets)};
    }

    inline void reorderProjections(Projections& proj,
                                    int num_proj, int num_nodes,
                                    int selected_features_count,
                                    int reorder_strategy, bool verbose) {
        if (reorder_strategy <= 0) return;

        int K = num_proj * selected_features_count;
        std::vector<std::vector<int>> rows(num_nodes, std::vector<int>(K));
        for (int n = 0; n < num_nodes; ++n)
            for (int p = 0; p < num_proj; ++p)
                for (int i = 0; i < selected_features_count; ++i)
                    rows[n][p * selected_features_count + i] = proj.col_idx[n][p][i];

        auto optimized = mcg::maximize_column_groupings(rows, reorder_strategy, verbose);

        for (int n = 0; n < num_nodes; ++n) {
            struct CW { int col; float w; };
            std::vector<CW> orig;
            orig.reserve(K);
            for (int p = 0; p < num_proj; ++p)
                for (int i = 0; i < selected_features_count; ++i)
                    orig.push_back({proj.col_idx[n][p][i],
                                    proj.weights[n][p][i]});

            std::vector<bool> used(orig.size(), false);
            for (int slot = 0; slot < K; ++slot) {
                int target = optimized[n][slot];
                bool found = false;
                for (size_t j = 0; j < orig.size(); ++j) {
                    if (!used[j] && orig[j].col == target) {
                        int p = slot / selected_features_count;
                        int i = slot % selected_features_count;
                        proj.col_idx[n][p][i] = orig[j].col;
                        proj.weights[n][p][i] = orig[j].w;
                        used[j] = true;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    int p = slot / selected_features_count;
                    int i = slot % selected_features_count;
                    proj.col_idx[n][p][i] = target;
                    proj.weights[n][p][i] = 1.0f;
                }
            }
        }
    }

    // Overload for AoS projection layout (e.g. YDF's
    // std::vector<std::vector<Projection>> where each Projection element
    // has .attribute_idx and .weight).
    template <typename Proj>
    inline void reorderProjections(std::vector<std::vector<Proj>>& all_node_projs,
                                    int num_proj, int num_nodes,
                                    int selected_features_count,
                                    int reorder_strategy, bool verbose) {
        if (reorder_strategy <= 0) return;

        int K = num_proj * selected_features_count;
        std::vector<std::vector<int>> rows(num_nodes, std::vector<int>(K));
        for (int n = 0; n < num_nodes; ++n)
            for (int p = 0; p < num_proj; ++p)
                for (int i = 0; i < selected_features_count; ++i)
                    rows[n][p * selected_features_count + i] =
                        all_node_projs[n][p][i].attribute_idx;

        auto optimized = mcg::maximize_column_groupings(rows, reorder_strategy, verbose);

        for (int n = 0; n < num_nodes; ++n) {
            struct CW { int col; float w; };
            std::vector<CW> orig;
            orig.reserve(K);
            for (int p = 0; p < num_proj; ++p)
                for (int i = 0; i < selected_features_count; ++i)
                    orig.push_back({all_node_projs[n][p][i].attribute_idx,
                                    all_node_projs[n][p][i].weight});

            std::vector<bool> used(orig.size(), false);
            for (int slot = 0; slot < K; ++slot) {
                int target = optimized[n][slot];
                bool found = false;
                for (size_t j = 0; j < orig.size(); ++j) {
                    if (!used[j] && orig[j].col == target) {
                        int p = slot / selected_features_count;
                        int i = slot % selected_features_count;
                        all_node_projs[n][p][i].attribute_idx = orig[j].col;
                        all_node_projs[n][p][i].weight = orig[j].w;
                        used[j] = true;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    int p = slot / selected_features_count;
                    int i = slot % selected_features_count;
                    all_node_projs[n][p][i].attribute_idx = target;
                    all_node_projs[n][p][i].weight = 1.0f;
                }
            }
        }
    }

    template <typename DataT>
    inline bool verifyProjection(const float* h_projected,
                                 const Projections& proj,
                                 const std::vector<DataT>& flattened,
                                 const std::vector<int>& selected_examples,
                                 const std::vector<int>& node_row_off,
                                 int num_proj, int num_nodes,
                                 int num_rows, int selected_features_count) {
        int total_selected = static_cast<int>(selected_examples.size());
        std::size_t proj_size = static_cast<std::size_t>(total_selected) * num_proj;
        std::vector<float> cpu_projected(proj_size, 0.0f);

        for (int p = 0; p < num_proj; ++p) {
            for (int n = 0; n < num_nodes; ++n) {
                int row_start = node_row_off[n];
                int rows_node = node_row_off[n + 1] - row_start;
                int base = row_start * num_proj + p * rows_node;
                for (int r = 0; r < rows_node; ++r) {
                    int ex_idx = selected_examples[row_start + r];
                    float sum = 0.0f;
                    for (int i = 0; i < selected_features_count; ++i) {
                        int col = proj.col_idx[n][p][i];
                        float w = proj.weights[n][p][i];
                        sum += w * flattened[static_cast<std::size_t>(col) * num_rows + ex_idx];
                    }
                    cpu_projected[base + r] = sum;
                }
            }
        }

        int mismatches = 0;
        for (std::size_t i = 0; i < proj_size; ++i) {
            float diff = std::abs(cpu_projected[i] - h_projected[i]);
            if (diff > 1e-5f) {
                if (mismatches < 10)
                    std::cout << "  MISMATCH [" << i << "]: cpu=" << cpu_projected[i]
                              << " gpu=" << h_projected[i] << " diff=" << diff << "\n";
                ++mismatches;
            }
        }
        if (mismatches == 0)
            std::cout << "CPU vs GPU check PASSED (" << proj_size << " values)\n";
        else
            std::cout << "CPU vs GPU check FAILED: " << mismatches << " mismatches\n";
        return mismatches == 0;
    }

}
