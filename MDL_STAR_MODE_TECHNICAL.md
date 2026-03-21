# MDL-Optimal Recursive Clustering Engine (Star Mode)

## Clustering Modes Overview

PyHercules offers three clustering modes for hierarchical K-means:

| Mode          | Description                         | Use Case                     |
| ------------- | ----------------------------------- | ---------------------------- |
| **Fixed K**   | User specifies K per level          | Known structure requirements |
| **Auto K**    | Silhouette/DB/CH score optimization | Exploratory analysis         |
| **Star Mode** | MDL-based two-level hierarchy       | Parameter-free segmentation  |

### Fixed K Mode

```python
hercules = Hercules(level_cluster_counts=[5, 3])  # 5 L1 clusters, 3 L2 clusters
```

### Auto K Mode

```python
hercules = Hercules(
    level_cluster_counts='auto',
    auto_k_method='silhouette',  # Options: 'silhouette', 'davies_bouldin', 'calinski_harabasz'
    auto_k_max=15
)
```

**Algorithm:** Tests K from `min_clusters_per_level` to `auto_k_max`, selects K maximizing/minimizing score:

- **Silhouette Score:** Maximize (higher = better separation)
- **Davies-Bouldin Index:** Minimize (lower = more compact clusters)
- **Calinski-Harabasz Index:** Maximize (higher = denser, well-separated clusters)

### Star Mode

```python
hercules = Hercules(level_cluster_counts='star')  # Zero parameters required
```

**Algorithm:** MDL-based Level 1 + business-driven Level 2 (details below)

---

## Star Mode: Technical Architecture

Star Mode implements a two-level hierarchical clustering with automatic K determination at both levels, specifically designed for unsupervised segmentation with zero hyperparameters.

### Level 1: K\* Means with MDL Principle

**Objective:** Discover optimal fine-grained micro-segments using Minimum Description Length

#### Algorithm: Iterative Split-Merge

```
Initialize: K=1, single centroid μ = mean(X)

Repeat until convergence (patience iterations without improvement):
  1. K-Means Step: Assign points to centroids, update centroids
  2. Sub-centroid Step: For each cluster, maintain 2 sub-centroids
  3. Split Evaluation:
     - For each cluster i with sub-centroids (μi1, μi2):
       - Calculate cost change if split into 2 clusters
       - Cost = residual_error + index_cost + model_cost
     - If best cost_change < 0: Split cluster, K++
  4. Merge Evaluation (if no split):
     - Find closest centroid pair
     - Calculate cost change if merged
     - If cost_change < 0: Merge clusters, K--
  5. Update best cost if improved, else increment unimproved_count

Convergence: unimproved_count >= patience
```

#### MDL Cost Function

```
MDL_Cost = Model_Cost + Index_Cost + Residual_Cost

Model_Cost = K × d × (max(X) - min(X)) / float_precision
Index_Cost = n × log(K)
Residual_Cost = (n × d × log(2π) + Σ||xi - μc(i)||²) / 2

where:
  K = number of clusters
  d = data dimensionality
  n = number of data points
  c(i) = cluster assignment for point i
```

**Key Properties:**

- **Model Cost:** Penalizes complexity (more clusters = higher cost)
- **Index Cost:** Encoding cost for cluster assignments
- **Residual Cost:** Within-cluster variance (reconstruction error)
- **Trade-off:** Balances cluster granularity vs. model simplicity

**Implementation:** `KStarMeans` class in `pyhercules.py` (lines 267-520)

**Typical Output:** K = 50-150 micro-segments (data-dependent)

---

### Level 2: Business-Driven Weighted K-Means

**Objective:** Group Level 1 micro-segments into actionable macro-segments

#### Algorithm: Threshold-Based K Selection

```
Input:
  - Level 1 centroids: {μ1, μ2, ..., μK1}
  - Level 1 sizes: {n1, n2, ..., nK1}
  - min_threshold_pct: Minimum population per L2 cluster (default 10%)

Initialize: K_candidate = min(7, K1)  # Heuristic starting point

For K = K_candidate down to 2:
  1. Run weighted K-Means on Level 1 centroids:
     - Features: μi (centroid vectors)
     - Weights: ni (population sizes)

  2. Calculate population percentages for each L2 cluster:
     percentage_j = (Σ ni for i in cluster_j) / (Σ all ni)

  3. Check threshold:
     min_percentage = min(percentage_j)

     If min_percentage >= min_threshold_pct:
       ✅ ACCEPT K and terminate
     Else:
       ❌ REJECT K, try K-1

Fallback: If no K meets threshold, use K=2

Output: Optimal K for Level 2
```

**Key Features:**

1. **Population Weighting:** Level 1 clusters vote proportionally to their size
   - Large micro-segments have more influence on Level 2 centroids
   - Prevents tiny outlier clusters from distorting macro-structure

2. **Business Constraint:** `min_threshold_pct` ensures actionable segments
   - 10% threshold → Each L2 cluster contains ≥10% of total population
   - Prevents unusable micro-segments at macro level

3. **Greedy Search:** Top-down K selection (high to low)
   - Starts with business-reasonable K (often 5-7 segments)
   - Reduces K until all clusters meet threshold
   - Maximizes granularity while respecting constraints

**Implementation:** `find_optimal_level2_k_star()` in `pyhercules.py` (lines 525-625)

**Typical Output:** K = 2-7 macro-segments

---

## Technical Comparison

### Auto K vs. Star Mode

| Aspect          | Auto K (Silhouette)           | Star Mode                       |
| --------------- | ----------------------------- | ------------------------------- |
| **Principle**   | Cluster cohesion/separation   | Information theory (MDL)        |
| **K Range**     | User-defined max              | Data-driven (unlimited)         |
| **Computation** | O(n × K_max) KMeans runs      | Iterative split/merge           |
| **Levels**      | Any depth                     | Fixed 2 levels                  |
| **Parameters**  | `auto_k_max`, `auto_k_method` | `star_mode_patience` (optional) |
| **Output**      | Similar K at each level       | L1 >> L2 (micro → macro)        |

### When to Use Star Mode

**Advantages:**

- ✅ Zero hyperparameter tuning (fully automatic)
- ✅ Theoretically grounded (MDL optimal for L1)
- ✅ Business-actionable segments (threshold constraint for L2)
- ✅ Hierarchical interpretability (drill-down from macro to micro)

**Limitations:**

- ❌ Fixed 2-level structure (cannot extend to L3+)
- ❌ Slower than fixed K (iterative search at L1)
- ❌ Requires numeric features (MDL assumes continuous space)

---

## Configuration Parameters

### Star Mode Initialization

```python
hercules = Hercules(
    level_cluster_counts='star',
    star_mode_min_threshold_pct=10.0,  # L2 minimum population % (range: 5-20)
    star_mode_patience=5,               # L1 convergence patience (iterations)
    random_state=42
)
```

#### `star_mode_min_threshold_pct`

- **Default:** 10.0
- **Range:** 5.0 - 20.0
- **Effect:**
  - Lower → More L2 clusters (finer macro-segmentation)
  - Higher → Fewer L2 clusters (coarser macro-segmentation)

#### `star_mode_patience`

- **Default:** 5
- **Range:** 3 - 10 (practical)
- **Effect:**
  - Lower → Faster convergence, may miss optimal K
  - Higher → More thorough search, slower runtime

---

## Computational Complexity

### Level 1 (K\* Means)

- **Per iteration:** O(n × d × K) for assignment + O(K² × d) for merge check
- **Total iterations:** Typically 20-100 (patience-dependent)
- **Overall:** O(I × n × d × K) where I = iteration count, K grows dynamically

### Level 2 (Weighted K-Means)

- **Per K candidate:** O(K1 × d × K2 × T) where T = KMeans iterations
- **K candidates tested:** ~5-7 (from initial_k down)
- **Overall:** O(K1 × d × K_init × T)

**Total:** Dominated by Level 1 iterative process. Expect 2-10x runtime vs. fixed K.

---

## Implementation Details

### Key Classes & Functions

| Component                      | Location                    | Purpose                               |
| ------------------------------ | --------------------------- | ------------------------------------- |
| `KStarMeans`                   | `pyhercules.py` L267-520    | MDL-based clustering with split/merge |
| `find_optimal_level2_k_star()` | `pyhercules.py` L525-625    | Business-driven L2 optimization       |
| Star mode dispatch             | `hercules.cluster()` method | Orchestrates 2-level star mode flow   |

### Algorithm Flow in `Hercules.cluster()`

```
1. Detect star mode: level_cluster_counts == 'star'

2. Level 1 Clustering:
   - Initialize KStarMeans(patience=star_mode_patience)
   - kstar.fit(scaled_data)
   - Extract labels, centroids → Create L1 Cluster objects

3. Level 2 Clustering:
   - Collect L1 centroids and sizes
   - Call find_optimal_level2_k_star(
       level1_centroids,
       level1_cluster_sizes,
       min_threshold_pct=star_mode_min_threshold_pct
     )
   - Assign L1 clusters to L2 based on weighted KMeans
   - Create L2 Cluster objects with L1 clusters as children

4. Return top-level L2 clusters (hierarchy complete)
```

---

## Example Output

```
Starting K*-means with patience=5
  Iteration 1: Split performed, K=2
  Iteration 5: Split performed, K=3
  Iteration 10: K=15, Cost=125432.45
  Iteration 15: Merge performed, K=14
  ...
  Iteration 47: K=87, Cost=112345.67

Converged after 47 iterations
Final K=87, Best Cost=112345.67

================================================================================
STAR MODE - LEVEL 2: Automatic K Selection
================================================================================
Level 1 K: 87 micro-segments
Total samples: 7,043
Minimum threshold: 10% (704 samples)
================================================================================

Starting K heuristic: min(7, 87) = 7

Trying K=7...
  Minimum cluster: 8.23%
  ❌ BELOW THRESHOLD

Trying K=6...
  Minimum cluster: 9.45%
  ❌ BELOW THRESHOLD

Trying K=5...
  Minimum cluster: 11.34%
  ✅ MEETS THRESHOLD

🎯 SUCCESS! Found optimal K=5

================================================================================
LEVEL 2 COMPLETE: Selected K=5
================================================================================

Star mode clustering complete!
  Level 1: 87 micro-segments (K* Means)
  Level 2: 5 macro-segments (Weighted Auto K)
```

---

## Summary

**Star Mode** = MDL-optimal micro-segmentation + business-constrained macro-segmentation

- **L1 (K\* Means):** Information-theoretic optimal K via split/merge with MDL cost
- **L2 (Weighted K-Means):** Population-balanced K via threshold-based search

**Result:** Fully automated two-level hierarchy requiring zero manual tuning, producing interpretable business segments with statistically optimal micro-structure.

**Technologies:** scikit-learn KMeans, custom MDL implementation, NumPy linear algebra
