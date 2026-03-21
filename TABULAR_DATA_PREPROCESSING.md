# Tabular Data Input & Pre-processing Layer

## Overview

PyHercules implements an automated preprocessing pipeline for tabular data:

1. **Detects** column types (numeric, categorical)
2. **Transforms** columns using appropriate techniques
3. **Standardizes** features for K-means clustering

---

## Stage 1: Column Type Detection

**Function:** `detect_column_types()` in `pyhercules_app.py` (lines 111-161)

Automatically classifies columns using rule-based heuristics:

### Numeric Columns

- Has numeric dtype (int, float) **OR**
- > 80% of values successfully convert to numeric via `pd.to_numeric(errors='coerce')`

### Categorical Columns

Both conditions must be met:

- Low cardinality: `(unique_values / total_values) < 0.05` **OR** `unique_values < 50`
- Short strings: `avg_length < 30 characters`

**Technologies:**

- `pandas.api.types.is_numeric_dtype()` - Type checking
- `pandas.to_numeric()` - Numeric conversion
- `pandas` DataFrame methods - Cardinality and string analysis

---

## Stage 2: Data Transformation

**Function:** `preprocess_mixed_data()` in `pyhercules_app.py` (lines 163-265)

### Numeric Processing

- Convert to numeric type (coerce errors to NaN)
- Fill missing values with column mean
- Output: 1 dimension per column

```python
col_data = pd.to_numeric(df[col], errors='coerce')
col_data = col_data.fillna(col_mean if not pd.isna(col_mean) else 0)
```

**Technologies:** `pandas.to_numeric()`, `pandas.fillna()`

### Categorical Processing

- Fill missing values with `'_MISSING_'` placeholder
- Apply one-hot encoding
- Output: N dimensions (N = unique categories)

```python
col_data = df[col].fillna('_MISSING_')
encoded = pd.get_dummies(col_data, prefix=col, drop_first=False, dtype=float)
```

**Technologies:** `pandas.get_dummies()`, `pandas.fillna()`

**Why One-Hot Encoding:**

- No ordinal assumptions about categories
- Compatible with K-means distance calculations
- Each category becomes independent binary feature

---

## Stage 3: Feature Consolidation

Combine processed columns into unified numerical array:

```python
combined_data = np.hstack(processed_parts)
```

Output format: `[numeric_col1, cat_col1_cat1, cat_col1_cat2, ..., numeric_col2, ...]`

**Technologies:** `numpy.hstack()`

---

## Stage 4: Standardization

**Function:** Internal to `Hercules.fit()` in `pyhercules.py` (lines 2046-2061)

Apply Z-score normalization:

```python
from sklearn.preprocessing import StandardScaler

self._scaler = StandardScaler()
standardized_data = self._scaler.fit_transform(numeric_data_unscaled)
```

**Formula:** `scaled_value = (value - mean) / std_deviation`

**Purpose:**

- Prevents large-scale features from dominating
- K-means uses Euclidean distance (scale-sensitive)
- Improves convergence speed

**Technologies:** `sklearn.preprocessing.StandardScaler`

**Note:** Original unscaled data preserved in `self.original_numeric_data_` for interpretable cluster statistics.

---

## Technical Flow

```
Raw Tabular Data (CSV/DataFrame)
         ↓
[Column Type Detection - pandas]
    • is_numeric_dtype()
    • to_numeric() conversion test
    • cardinality analysis
         ↓
┌─────────────────┬──────────────────┐
│ Numeric         │ Categorical      │
│ (pandas)        │ (pandas)         │
│ • to_numeric()  │ • get_dummies()  │
│ • fillna(mean)  │ • fillna('_MISSING_') │
└─────────────────┴──────────────────┘
         ↓
[Feature Concatenation - numpy.hstack()]
         ↓
[Combined Numerical Matrix]
         ↓
[StandardScaler - sklearn]
    • Z-score normalization
    • Preserve original for stats
         ↓
[Clustering-Ready Features]
         ↓
[K-Means Clustering - sklearn.cluster.KMeans]
```

---

## Technology Stack

| Component            | Library      | Methods                                                        |
| -------------------- | ------------ | -------------------------------------------------------------- |
| Type Detection       | pandas       | `is_numeric_dtype()`, `to_numeric()`, `nunique()`, `str.len()` |
| Numeric Processing   | pandas       | `to_numeric()`, `fillna()`                                     |
| Categorical Encoding | pandas       | `get_dummies()`, `fillna()`                                    |
| Feature Combination  | NumPy        | `hstack()`                                                     |
| Standardization      | scikit-learn | `StandardScaler.fit_transform()`                               |
| Clustering           | scikit-learn | `KMeans`                                                       |

---

## Metadata Preservation

Metadata tracked for each feature:

```python
# Numeric
{'name': 'Age', 'type': 'numeric', 'source_column': 'Age'}

# Categorical (encoded)
{'name': 'Country_USA', 'type': 'categorical_encoded',
 'source_column': 'Country', 'category': 'USA'}
```

Enables feature interpretation and traceability in cluster descriptions.

---

## Summary

**Pipeline:** Column Detection → Type-Specific Transformation → Consolidation → Standardization → Clustering

**Core Technologies:** pandas (preprocessing), NumPy (arrays), scikit-learn (scaling & clustering)

**Key Features:** Automatic type detection, one-hot encoding for categorical data, Z-score normalization, metadata preservation for interpretability.
