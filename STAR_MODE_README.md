# Star Mode - K\* Means Hierarchical Clustering

## Overview

**Star Mode** is a parameter-free two-level hierarchical clustering algorithm integrated into PyHercules. It automatically determines the optimal number of clusters at both levels, requiring **zero manual parameter tuning**.

## What is Star Mode?

Star Mode combines two powerful automatic clustering techniques:

### Level 1: K\* Means (Micro-Segmentation)

- Uses the **Minimum Description Length (MDL) principle** to automatically find optimal K
- Iteratively splits and merges clusters based on cost reduction
- Creates fine-grained micro-segments for detailed analysis
- No manual K needed—algorithm converges automatically

### Level 2: Weighted K-Means with Business-Driven Auto K (Macro-Segmentation)

- Takes Level 1 micro-segments and groups them into actionable macro-segments
- **Automatic K selection** with business constraints:
  - Ensures all clusters have ≥10% of population (configurable)
  - Uses population weighting for balanced segments
  - Searches from high K down to K=2 until threshold is met
- Guarantees business-actionable segments (no tiny segments)

## How to Use Star Mode

### In the PyHercules App:

1. Upload your data as usual
2. In the **"Cluster Counts/Level"** field, enter: `star`
3. Click "Run Clustering"
4. The algorithm will automatically:
   - Determine optimal K for Level 1 (typically 50-100+ micro-segments)
   - Determine optimal K for Level 2 (typically 2-7 macro-segments)
   - Create a two-level hierarchy you can explore

### In Python Code:

```python
from pyhercules import Hercules

# Initialize with 'star' mode
hercules = Hercules(
    level_cluster_counts='star',  # Activate star mode
    representation_mode='direct',
    random_state=42
)

# Run clustering (will automatically determine K at both levels)
top_clusters = hercules.cluster(your_numeric_data)

# Results will have exactly 2 levels:
# Level 1: Many micro-segments (K determined by MDL)
# Level 2: Few macro-segments (K determined by threshold)
print(f"Level 1: {len([c for c in hercules._all_clusters_map.values() if c.level == 1])} micro-segments")
print(f"Level 2: {len([c for c in hercules._all_clusters_map.values() if c.level == 2])} macro-segments")
```

## When to Use Star Mode

**✅ Use Star Mode when:**

- You don't know how many clusters you need
- You want fully automatic clustering with no parameter tuning
- You need both strategic overview (Level 2) and tactical detail (Level 1)
- You want business-actionable segments (≥10% population each)
- Working with customer segmentation, market analysis, or similar use cases

**❌ Don't use Star Mode when:**

- You have specific requirements for exact number of clusters at multiple levels
- You need more than 2 levels of hierarchy
- Working with non-numeric data (images, text) - star mode is optimized for numeric features

## Star Mode vs. Regular Modes

| Feature                  | Fixed Mode           | Auto Mode            | Star Mode             |
| ------------------------ | -------------------- | -------------------- | --------------------- |
| **K Selection**          | Manual               | Silhouette/DB/CH     | MDL + Threshold       |
| **Levels**               | As many as specified | As many as needed    | Always 2              |
| **Best For**             | Known structure      | Exploratory          | Business segmentation |
| **Parameters**           | Many (K per level)   | Some (max K, metric) | Zero                  |
| **Business Constraints** | No                   | No                   | Yes (threshold)       |

## Advanced Configuration

You can customize star mode behavior when initializing Hercules:

```python
hercules = Hercules(
    level_cluster_counts='star',
    star_mode_min_threshold_pct=15.0,  # Require 15% minimum per Level 2 cluster
    star_mode_patience=5,              # K* Means convergence patience
    random_state=42
)
```

### Parameters:

- **`star_mode_min_threshold_pct`** (default: 10.0)
  - Minimum percentage of population per Level 2 cluster
  - Range: 5.0 to 20.0
  - Higher = fewer, larger segments
  - Lower = more, smaller segments

- **`star_mode_patience`** (default: 5)
  - Number of iterations without MDL improvement before K\* Means stops
  - Higher = more thorough search (slower)
  - Lower = faster convergence

## Example Output

```
Starting K*-means with patience=5
  Iteration 1: Split performed, K=2
  Iteration 5: Split performed, K=3
  Iteration 10: K=15, Cost=125432.45
  ...
Converged after 47 iterations
Final K=94, Best Cost=112345.67

STAR MODE - LEVEL 2: Automatic K Selection
Level 1 K: 94 micro-segments
Total samples: 7,043
Minimum threshold: 10% (704 samples)

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

LEVEL 2 COMPLETE: Selected K=5

Star mode clustering complete!
  Level 1: 94 micro-segments (K* Means)
  Level 2: 5 macro-segments (Weighted Auto K)
```

## Implementation Details

Star Mode is based on the research paper implementing K\* Means with MDL-based split/merge operations. The Level 2 selection algorithm uses a novel business-driven approach that:

1. Starts with a sensible heuristic: `min(7, K_Level1)`
2. Performs weighted K-means on Level 1 centroids (using population as weights)
3. Checks if all clusters meet minimum threshold
4. Iteratively reduces K until threshold is satisfied
5. Falls back to K=2 if no K meets threshold

### Key Advantages:

- **Zero hyperparameters** for clustering structure
- **Deterministic results** (with fixed random_state)
- **Business-driven** (ensures actionable segments)
- **Hierarchical flexibility** (drill-down from macro to micro)

## Troubleshooting

**Q: What if my Level 2 doesn't meet the threshold?**
A: The algorithm will fall back to K=2 as a best-effort. Consider lowering `star_mode_min_threshold_pct` if you need more granular Level 2 segments.

**Q: Can I use star mode with text or image data?**
A: Star mode works with any numeric feature vectors. For text/images, the embeddings will be treated as numeric features.

**Q: Star mode is slow. How can I speed it up?**
A: Reduce `star_mode_patience` from 5 to 3. This makes K\* Means converge faster but may find a suboptimal K.

**Q: Can I have more than 2 levels in star mode?**
A: No, star mode is specifically designed for two-level hierarchy. Use regular auto mode for multiple levels.

## References

- Based on implementation from the "7th exploration absolute zscore features" notebook in the PyHercules dataset

## License

Star Mode is part of PyHercules and follows the same license.
