#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_MAXIMIZE_GROUPINGS_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_MAXIMIZE_GROUPINGS_H_

#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <map>
#include <random>
#include <cmath>
#include <climits>
#include <limits>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <functional>

namespace mcg {

using Item  = int;
using Row   = std::vector<Item>;
using Table = std::vector<Row>;

inline Table order_rows_for_locality(const Table& t);

inline std::vector<int> hungarian(const std::vector<std::vector<double>>& cost) {
    int n = (int)cost.size();
    std::vector<double> u(n+1), v(n+1);
    std::vector<int> p(n+1), way(n+1);
    for (int i = 1; i <= n; i++) {
        p[0] = i; int j0 = 0;
        std::vector<double> minv(n+1, 1e18);
        std::vector<bool> used(n+1, false);
        do {
            used[j0] = true; int i0 = p[j0], j1 = 0; double delta = 1e18;
            for (int j = 1; j <= n; j++) {
                if (!used[j]) {
                    double cur = cost[i0-1][j-1] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            }
            for (int j = 0; j <= n; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    std::vector<int> assignment(n);
    for (int j = 1; j <= n; j++) assignment[p[j]-1] = j-1;
    return assignment;
}

inline long long grouping_pairs(const Table& t) {
    if (t.empty()) return 0;
    int N = (int)t.size(), K = (int)t[0].size();
    long long score = 0;
    for (int j = 0; j < K; ++j) {
        std::unordered_map<int,int> cnt;
        for (int i = 0; i < N; ++i) ++cnt[t[i][j]];
        for (auto& [v,c] : cnt) score += (long long)c*(c-1)/2;
    }
    return score;
}

inline int grouped_cells(const Table& t) {
    if (t.empty()) return 0;
    int N = (int)t.size(), K = (int)t[0].size(), total = 0;
    for (int j = 0; j < K; ++j) {
        std::unordered_map<int,int> cnt;
        for (int i = 0; i < N; ++i) ++cnt[t[i][j]];
        for (auto& [v,c] : cnt) if (c > 1) total += c;
    }
    return total;
}

inline int uniform_columns(const Table& t) {
    if (t.empty()) return 0;
    int N = (int)t.size(), K = (int)t[0].size(), cnt = 0;
    for (int j = 0; j < K; ++j) {
        bool same = true;
        for (int i = 1; i < N; ++i) if (t[i][j] != t[0][j]) { same = false; break; }
        if (same) ++cnt;
    }
    return cnt;
}

inline int locality_score(const Table& t) {
    if (t.size() < 2) return 0;
    int N = (int)t.size(), K = (int)t[0].size(), score = 0;
    for (int s = 0; s+1 < N; ++s)
        for (int j = 0; j < K; ++j)
            if (t[s][j] == t[s+1][j]) ++score;
    return score;
}

inline long long upper_bound(const Table& t) {
    if (t.empty()) return 0;
    int N = (int)t.size();
    std::unordered_map<int,int> rp;
    for (int n = 0; n < N; ++n) {
        std::unordered_map<int,bool> seen;
        for (int v : t[n]) if (!seen[v]) { ++rp[v]; seen[v] = true; }
    }
    long long bound = 0;
    for (auto& [v,f] : rp) { long long g = std::min(f,N); bound += g*(g-1)/2; }
    return bound;
}

struct ScoreSummary { long long pairs; int cells, uniform, locality; };

inline ScoreSummary score_all(const Table& t) {
    return { grouping_pairs(t), grouped_cells(t), uniform_columns(t), locality_score(t) };
}

inline void print_scores(const char* label, const ScoreSummary& s, int K) {
    std::cout << "  " << std::left << std::setw(28) << label
              << "pairs=" << std::setw(6) << s.pairs
              << "  grouped_cells=" << std::setw(6) << s.cells
              << "  uniform=" << s.uniform << "/" << K
              << "  locality=" << s.locality << "\n";
}

inline void print_table(const Table& t, const char* label) {
    int N = (int)t.size(), K = (int)t[0].size();
    std::cout << "\n  " << label << "  (" << N << " rows x " << K << " cols)\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "    row " << std::setw(3) << i << ":";
        for (int j = 0; j < K; ++j) std::cout << " " << std::setw(3) << t[i][j];
        std::cout << "\n";
    }
    std::cout << "\n  Column groups:\n";
    for (int j = 0; j < K; ++j) {
        std::map<int,int> groups;
        for (int i = 0; i < N; ++i) ++groups[t[i][j]];
        std::cout << "    col " << std::setw(2) << j << ":";
        for (int i = 0; i < N; ++i) std::cout << " " << std::setw(3) << t[i][j];
        std::cout << "  |";
        for (auto& [v,c] : groups) if (c > 1) std::cout << " " << v << "x" << c;
        std::cout << "\n";
    }
}

// Strategy 0: Hungarian dissimilarity alignment (align_sets + group_for_locality)
// Minimises Σ|a-b| across all row pairs at each position via sequential
// Hungarian assignment.  This captures "close but not identical" alignment
// that exact-match strategies miss.
inline Table hungarian_dissimilarity(const Table& rows) {
    int N = (int)rows.size(), K = (int)rows[0].size();

    auto do_align = [&](int ref_idx) -> Table {
        Table sets = rows;
        std::swap(sets[0], sets[ref_idx]);
        std::sort(sets[0].begin(), sets[0].end());
        for (int s = 1; s < N; ++s) {
            std::vector<std::vector<double>> cost(K, std::vector<double>(K, 0.0));
            for (int pos = 0; pos < K; ++pos)
                for (int elem = 0; elem < K; ++elem)
                    for (int prev = 0; prev < s; ++prev)
                        cost[pos][elem] += std::abs(sets[s][elem] - sets[prev][pos]);
            auto assignment = hungarian(cost);
            Row reordered(K);
            for (int pos = 0; pos < K; ++pos)
                reordered[pos] = sets[s][assignment[pos]];
            sets[s] = reordered;
        }
        return sets;
    };

    auto dissimilarity = [&](const Table& t) -> double {
        double total = 0;
        for (int pos = 0; pos < K; ++pos)
            for (int i = 0; i < N; ++i)
                for (int j = i + 1; j < N; ++j)
                    total += std::abs(t[i][pos] - t[j][pos]);
        return total;
    };

    double best_d = 1e18;
    Table best;
    for (int ref = 0; ref < N; ++ref) {
        Table result = do_align(ref);
        double d = dissimilarity(result);
        if (d < best_d) { best_d = d; best = result; }
    }
    return best;
}

// Strategy 1: Greedy most-frequent column packing
inline Table greedy_most_frequent(const Table& rows) {
    int N = (int)rows.size(), K = (int)rows[0].size();
    Table result(N, Row(K));
    std::vector<Row> pending(N);
    for (int i = 0; i < N; ++i) pending[i] = rows[i];
    std::vector<int> next_pos(N, 0);
    int items_left = N * K;
    while (items_left > 0) {
        std::unordered_map<int,int> freq;
        for (auto& r : pending) for (int x : r) ++freq[x];
        int best_val = 0, best_cnt = 0;
        for (auto& [v,c] : freq)
            if (c > best_cnt || (c == best_cnt && v > best_val)) { best_val = v; best_cnt = c; }
        for (int r = 0; r < N; ++r) {
            auto it = std::find(pending[r].begin(), pending[r].end(), best_val);
            if (it != pending[r].end()) { result[r][next_pos[r]++] = best_val; pending[r].erase(it); --items_left; }
        }
    }
    return result;
}

// Strategy 2: Column auction
inline Table column_auction(const Table& rows) {
    int N = (int)rows.size(), K = (int)rows[0].size();
    Table result(N, Row(K, -1));
    std::vector<Row> remaining(N);
    for (int i = 0; i < N; ++i) remaining[i] = rows[i];
    std::vector<bool> pos_done(K, false);
    int placed = 0, target = N * K;
    while (placed < target) {
        int best_val = -1, best_pos = -1, best_count = 0;
        for (int j = 0; j < K; ++j) {
            if (pos_done[j]) continue;
            std::unordered_map<int,int> can_supply;
            for (int n = 0; n < N; ++n) {
                if (result[n][j] != -1) continue;
                for (int v : remaining[n]) ++can_supply[v];
            }
            for (auto it = can_supply.begin(); it != can_supply.end(); ++it) {
                int v = it->first, cnt = 0;
                for (int n = 0; n < N; ++n) {
                    if (result[n][j] != -1) continue;
                    for (int x : remaining[n]) if (x == v) { ++cnt; break; }
                }
                it->second = cnt;
            }
            for (auto& [v,c] : can_supply)
                if (c > best_count || (c == best_count && v > best_val)) { best_count = c; best_val = v; best_pos = j; }
        }
        if (best_pos < 0 || best_count <= 0) break;
        for (int n = 0; n < N; ++n) {
            if (result[n][best_pos] != -1) continue;
            auto it = std::find(remaining[n].begin(), remaining[n].end(), best_val);
            if (it != remaining[n].end()) { result[n][best_pos] = best_val; remaining[n].erase(it); ++placed; }
        }
        bool all_filled = true;
        for (int n = 0; n < N; ++n) if (result[n][best_pos] == -1) { all_filled = false; break; }
        if (all_filled) pos_done[best_pos] = true;
        if (best_count <= 1) break;
    }
    for (int n = 0; n < N; ++n)
        for (int j = 0; j < K; ++j)
            if (result[n][j] == -1 && !remaining[n].empty()) { result[n][j] = remaining[n].back(); remaining[n].pop_back(); }
    return result;
}

// Strategy 3: Iterative Hungarian alignment
// reorder_nodes: after converging, reorder rows to maximise adjacent-row locality.
inline Table iterative_hungarian_grouping(const Table& rows, int max_iters = 30, bool reorder_nodes = false) {
    int N = (int)rows.size(), K = (int)rows[0].size();
    Table current = greedy_most_frequent(rows);
    long long best_score = grouping_pairs(current);
    Table best = current;
    std::vector<std::unordered_map<int,int>> col_cnt(K);
    for (int j = 0; j < K; ++j) for (int n = 0; n < N; ++n) ++col_cnt[j][current[n][j]];
    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;
        for (int n = 0; n < N; ++n) {
            for (int j = 0; j < K; ++j) { int v = current[n][j]; if (--col_cnt[j][v] == 0) col_cnt[j].erase(v); }
            Row elements = rows[n];
            std::vector<std::vector<double>> cost(K, std::vector<double>(K, 0.0));
            for (int ei = 0; ei < K; ++ei) { int ev = elements[ei]; for (int pos = 0; pos < K; ++pos) { auto it = col_cnt[pos].find(ev); cost[ei][pos] = -(double)(it != col_cnt[pos].end() ? it->second : 0); } }
            auto assignment = hungarian(cost);
            Row new_row(K);
            for (int ei = 0; ei < K; ++ei) new_row[assignment[ei]] = elements[ei];
            if (new_row != current[n]) changed = true;
            current[n] = new_row;
            for (int j = 0; j < K; ++j) ++col_cnt[j][current[n][j]];
        }
        long long sc = grouping_pairs(current);
        if (sc > best_score) { best_score = sc; best = current; }
        if (!changed) break;
    }
    if (reorder_nodes) best = order_rows_for_locality(best);
    return best;
}

// Strategy 4: Simulated annealing (O(1) delta per swap)
inline Table simulated_annealing_grouping(const Table& rows, double T_start = 50.0, double T_end = 0.005, double cooling = 0.997, int steps_per_temp = 0, unsigned seed = 42) {
    int N = (int)rows.size(), K = (int)rows[0].size();
    if (steps_per_temp <= 0) steps_per_temp = std::max(N * K * 3, 500);
    Table current = greedy_most_frequent(rows);
    long long cur_score = grouping_pairs(current), best_score = cur_score;
    Table best = current;
    std::vector<std::unordered_map<int,int>> col_cnt(K);
    for (int j = 0; j < K; ++j) for (int n = 0; n < N; ++n) ++col_cnt[j][current[n][j]];
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> row_dist(0, N-1), col_dist(0, K-1);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    auto pairs = [](int c) -> long long { return (long long)c*(c-1)/2; };
    double T = T_start;
    while (T > T_end) {
        for (int step = 0; step < steps_per_temp; ++step) {
            int r = row_dist(rng), i = col_dist(rng), j = col_dist(rng);
            if (i == j) continue;
            int vi = current[r][i], vj = current[r][j];
            if (vi == vj) continue;
            int ci_vi = col_cnt[i][vi], ci_vj = col_cnt[i].count(vj) ? col_cnt[i][vj] : 0;
            int cj_vi = col_cnt[j].count(vi) ? col_cnt[j][vi] : 0, cj_vj = col_cnt[j][vj];
            long long old_p = pairs(ci_vi)+pairs(ci_vj)+pairs(cj_vi)+pairs(cj_vj);
            long long new_p = pairs(ci_vi-1)+pairs(ci_vj+1)+pairs(cj_vi+1)+pairs(cj_vj-1);
            long long delta = new_p - old_p;
            if (delta > 0 || uni(rng) < std::exp((double)delta / T)) {
                current[r][i] = vj; current[r][j] = vi;
                col_cnt[i][vi]--; if (!col_cnt[i][vi]) col_cnt[i].erase(vi); col_cnt[i][vj]++;
                col_cnt[j][vj]--; if (!col_cnt[j][vj]) col_cnt[j].erase(vj); col_cnt[j][vi]++;
                cur_score += delta;
                if (cur_score > best_score) { best_score = cur_score; best = current; }
            }
        }
        T *= cooling;
    }
    return best;
}

inline Table multi_start_sa(const Table& rows, int starts = 5) {
    Table best; long long best_score = -1;
    for (int i = 0; i < starts; ++i) {
        Table result = simulated_annealing_grouping(rows, 50.0, 0.005, 0.997, 0, 42+i*997);
        long long s = grouping_pairs(result);
        if (s > best_score) { best_score = s; best = result; }
    }
    return best;
}

// Strategy 6: Symmetric — every node uses the same column indices per slot.
// Picks the K most popular column indices across all nodes and replicates
// that single assignment to every row, giving perfect column uniformity.
inline Table symmetric_uniform(const Table& rows) {
    int N = (int)rows.size(), K = (int)rows[0].size();

    std::unordered_map<int,int> value_freq;
    for (int n = 0; n < N; ++n) {
        std::unordered_map<int,bool> seen;
        for (int v : rows[n])
            if (!seen[v]) { ++value_freq[v]; seen[v] = true; }
    }

    std::vector<std::pair<int,int>> freq_list(value_freq.begin(), value_freq.end());
    std::sort(freq_list.begin(), freq_list.end(), [](auto& a, auto& b) {
        return a.second > b.second || (a.second == b.second && a.first < b.first);
    });

    Row consensus(K);
    for (int j = 0; j < K; ++j)
        consensus[j] = (j < (int)freq_list.size()) ? freq_list[j].first : 0;

    return Table(N, consensus);
}

// Row ordering: maximise adjacent-row locality
inline int pairwise_matches(const Row& a, const Row& b) {
    int c = 0; for (size_t i = 0; i < a.size(); ++i) if (a[i] == b[i]) ++c; return c;
}

inline Table order_rows_for_locality(const Table& t) {
    int N = (int)t.size();
    if (N <= 1) return t;
    if (N <= 10) {
        std::vector<int> perm(N); std::iota(perm.begin(), perm.end(), 0);
        int best_loc = -1; std::vector<int> best_perm;
        do {
            Table reordered(N); for (int i = 0; i < N; ++i) reordered[i] = t[perm[i]];
            int loc = locality_score(reordered);
            if (loc > best_loc) { best_loc = loc; best_perm = perm; }
        } while (std::next_permutation(perm.begin(), perm.end()));
        Table result(N); for (int i = 0; i < N; ++i) result[i] = t[best_perm[i]]; return result;
    }
    int seed_a = 0, seed_b = 1, seed_score = -1;
    for (int i = 0; i < N; ++i) for (int j = i+1; j < N; ++j) {
        int m = pairwise_matches(t[i], t[j]);
        if (m > seed_score) { seed_score = m; seed_a = i; seed_b = j; }
    }
    std::vector<int> order = {seed_a, seed_b}; std::vector<bool> used(N, false);
    used[seed_a] = used[seed_b] = true;
    while ((int)order.size() < N) {
        int best_idx = -1, best_m = -1; bool at_front = false;
        for (int i = 0; i < N; ++i) {
            if (used[i]) continue;
            int mf = pairwise_matches(t[i], t[order.front()]), mb = pairwise_matches(t[i], t[order.back()]);
            if (mb >= mf && mb > best_m) { best_m = mb; best_idx = i; at_front = false; }
            if (mf > mb && mf > best_m) { best_m = mf; best_idx = i; at_front = true; }
        }
        used[best_idx] = true;
        if (at_front) order.insert(order.begin(), best_idx); else order.push_back(best_idx);
    }
    Table result(N); for (int i = 0; i < N; ++i) result[i] = t[order[i]]; return result;
}

// Strategy selector
//  0 = none (keep original)
//  1 = greedy  (fast, good)
//  2 = hungarian_reorder  (old align_sets approach)
//  3 = column_auction
//  4 = iterative_hungarian
//  5 = simulated_annealing_5x  (best quality, slow)
//  6 = symmetric  (all nodes same column indices, perfect uniformity)
//  7 = iterative_hungarian + node reordering for locality
inline const char* strategy_name(int id) {
    switch (id) {
        case 0: return "none";
        case 1: return "greedy";
        case 2: return "hungarian_reorder";
        case 3: return "column_auction";
        case 4: return "iterative_hungarian";
        case 5: return "simulated_annealing_5x";
        case 6: return "symmetric";
        case 7: return "iterative_hungarian+node_reorder";
        default: return "unknown";
    }
}

inline Table maximize_column_groupings(const Table& rows, int strategy, bool verbose = true) {
    int N = (int)rows.size(), K = (int)rows[0].size();
    if (strategy <= 0) return rows;

    auto orig = score_all(rows);
    long long ub = upper_bound(rows);

    auto t0 = std::chrono::steady_clock::now();
    Table result;
    switch (strategy) {
        case 1: result = greedy_most_frequent(rows); break;
        case 2: result = hungarian_dissimilarity(rows); break;
        case 3: result = column_auction(rows); break;
        case 4: result = iterative_hungarian_grouping(rows); break;
        case 5: result = multi_start_sa(rows, 5); break;
        case 6: result = symmetric_uniform(rows); break;
        case 7: result = iterative_hungarian_grouping(rows, 30, true); break;
        default: return rows;
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t0).count();
    auto sc = score_all(result);

    if (verbose) {
        std::cout << "\n══════════════════════════════════════════════\n"
                  << "  strategy " << strategy << ": " << strategy_name(strategy)
                  << "  (" << N << " rows x " << K << " cols, " << ms << " ms)\n";
        print_scores("Original", orig, K);
        print_scores("Result",   sc,   K);
        std::cout << "  pairs:    " << orig.pairs << " -> " << sc.pairs << "  (+" << sc.pairs-orig.pairs << ")\n"
                  << "  upper bound: " << ub;
        if (sc.pairs >= ub) std::cout << "  ACHIEVED"; else std::cout << "  (" << std::fixed << std::setprecision(1) << 100.0*sc.pairs/ub << "%)";
        std::cout << "\n══════════════════════════════════════════════\n";
    }
    return result;
}

} // namespace mcg
#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_MAXIMIZE_GROUPINGS_H_
