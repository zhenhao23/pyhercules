# Mixed Data Type Support Implementation

## Overview

This implementation adds support for **mixed data types** (numerical, categorical, and textual) in tabular datasets for the Hercules clustering application. Previously, only numerical columns were supported.

## Key Features

### 1. **Automatic Column Type Detection**

- **Numerical**: Detected using `pd.api.types.is_numeric_dtype()`
- **Categorical**: Low cardinality (< 5% unique values OR < 50 unique values) AND short text (< 30 chars average)
- **Text**: Everything else (long strings, free-form text)

**Detection Logic:**

```python
def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Automatically detects column types as 'numeric', 'categorical', or 'text'.

    Detection criteria:
    - Numeric: pd.api.types.is_numeric_dtype()
    - Categorical: cardinality < 5% OR unique_count < 50, AND avg_length < 30
    - Text: All other object/string columns
    """
```

### 2. **Mixed Data Preprocessing Pipeline**

Transforms all data types into a unified numerical representation:

**Preprocessing Steps:**

- **Numerical columns**: Kept as-is, scaled with StandardScaler
- **Categorical columns**: One-hot encoded (with `drop_first=True` to avoid collinearity)
- **Text columns**: TF-IDF vectorization → TruncatedSVD (dimensionality reduction to 10 components)
- **Final step**: StandardScaler applied to all combined features

**Function:**

```python
def preprocess_mixed_data(df, column_types, variable_metadata=None):
    """
    Returns: (scaled_array, column_names, metadata_dict)

    Metadata tracks:
    - Source column for each feature
    - Type of transformation applied
    - Additional info (category name, SVD dimension, etc.)
    """
```

### 3. **User Interface Enhancements**

**Auto-Detection with Manual Override:**

- UI displays detected type for each column with color-coded badges:
  - 📊 **Numeric** (blue badge)
  - 🏷️ **Categorical** (green badge)
  - 📝 **Text** (light blue badge)
- Dropdown allows users to override detection or ignore columns
- Options: Numeric, Categorical, Text, **Ignore**

**UI Layout:**

```
┌─────────────────────────────────────────────┐
│ Column Type Detection & Selection           │
├─────────────┬──────────┬────────────────────┤
│ Column Name │ Detected │ Override Dropdown  │
├─────────────┼──────────┼────────────────────┤
│ age         │ 📊 Numeric │ [Dropdown]       │
│ city        │ 🏷️ Categorical │ [Dropdown]   │
│ review_text │ 📝 Text  │ [Dropdown]         │
└─────────────┴──────────┴────────────────────┘
```

## Implementation Details

### Files Modified

- **`pyhercules_app.py`**: Main application file

### New Functions Added

1. **`detect_column_types(df)`**: Automatic type detection
2. **`preprocess_mixed_data(df, column_types, variable_metadata)`**: Unified preprocessing

### New Components

1. **`column-types-store`**: Stores detected/overridden column types
2. **`column-type-selector-container`**: UI for column type review/override
3. **`update_column_types` callback**: Handles manual overrides

### Updated Components

1. **`update_tabular_preview` callback**: Now detects column types and creates UI
2. **`initiate_run` callback**: Uses `column_types` instead of `selected_columns`
3. **`run_hercules_clustering` callback**: Applies mixed data preprocessing
4. **`go_back_to_upload` callback**: Clears column type store

## Usage Flow

### User Experience:

1. **Upload** tabular CSV/Excel file
2. **Review** auto-detected column types (shown with badges)
3. **Override** any misclassifications via dropdowns (optional)
4. **Run** clustering with mixed data automatically preprocessed

### Example Scenario:

```
Customer Dataset:
- age (numeric) → Kept as numeric
- income (numeric) → Kept as numeric
- city (categorical) → One-hot encoded to city_NYC, city_LA, city_SF
- feedback (text) → TF-IDF + SVD to 10 dimensions

Final feature space: 2 + 2 + 10 = 14 dimensions (all scaled)
```

## Technical Details

### TF-IDF Parameters:

```python
TfidfVectorizer(
    max_features=100,      # Top 100 words
    stop_words='english',  # Remove common words
    min_df=1,              # Minimum document frequency
    max_df=0.95           # Maximum document frequency
)
```

### SVD Reduction:

- Target: 10 components
- Adjusted automatically if fewer features exist
- Captures semantic meaning of text

### Metadata Tracking:

Each transformed feature includes:

```python
{
    'type': 'categorical_encoded' | 'text_embedding' | 'numeric',
    'source_column': 'original_column_name',
    'category': '...',        # For categorical
    'dimension': 0,           # For text embeddings
    'method': 'tfidf_svd'     # Transformation method
}
```

## Benefits

✅ **Automatic**: Works out-of-the-box with reasonable defaults  
✅ **Flexible**: Users can override any auto-detection  
✅ **Transparent**: Shows detection logic via badges and UI  
✅ **Educational**: Users learn about their data structure  
✅ **Robust**: Handles edge cases (empty columns, all-unique values, etc.)

## Edge Cases Handled

1. **Empty text columns**: Skipped in preprocessing
2. **High-cardinality categoricals**: Treated as text if cardinality too high
3. **Numeric-looking categories**: Requires manual override (e.g., year as category)
4. **All columns ignored**: Error message prevents empty clustering
5. **Mixed preprocessing failure**: Graceful fallback to numeric-only mode

## Performance Notes

- TF-IDF is efficient for moderate text sizes (< 10,000 docs recommended)
- One-hot encoding can create many features for high-cardinality categoricals
- StandardScaler ensures all feature types are on same scale
- SVD reduces text dimensionality to prevent feature explosion

## Future Enhancements

Possible improvements:

1. **Confidence scores** for auto-detection
2. **Customizable thresholds** (cardinality, text length)
3. **Alternative encodings** (target encoding, embeddings for categoricals)
4. **Advanced text models** (sentence transformers instead of TF-IDF)
5. **Feature importance** visualization post-clustering

## Dependencies

Required packages (already in requirements):

- `scikit-learn` (TfidfVectorizer, TruncatedSVD, StandardScaler)
- `pandas` (DataFrame operations)
- `numpy` (Array operations)

---

**Implementation Date**: October 27, 2025  
**Version**: 1.0  
**Status**: ✅ Complete and Ready for Testing
