# Actionable Summarization Layer

## Overview

The Actionable Summarization Layer transforms raw cluster structures into human-interpretable business insights through two key mechanisms:

1. **Feature-Driven Pre-Analysis:** Identifies and prioritizes the most defining features using statistical deviation
2. **Contextualized LLM Prompting:** Incorporates domain knowledge and business goals to generate relevant descriptions

This layer ensures LLM-generated cluster descriptions are both statistically grounded and business-actionable.

---

## 1. Feature-Driven Pre-Analysis

### Objective

Reduce noise and focus the LLM on the **most defining characteristics** of each cluster by filtering features based on their deviation from global patterns.

### Algorithm: Absolute Z-Score Ranking

#### Step 1: Calculate Deviation from Global Mean

For each cluster and each numeric feature:

```python
# Per-feature deviation calculation
cluster_mean = mean(cluster_data[:, feature_i])
global_mean = mean(all_data[:, feature_i])

# Percentage deviation (relative to global magnitude)
if abs(global_mean) > 1e-10:
    deviation_pct = ((cluster_mean - global_mean) / abs(global_mean)) * 100
else:
    deviation_pct = cluster_mean * 100  # Handle near-zero global means

# Absolute deviation (for ranking)
abs_deviation_pct = abs(deviation_pct)
```

**Interpretation:**

- **High absolute deviation:** Feature strongly differentiates this cluster from overall dataset
- **Low absolute deviation:** Feature similar to global pattern (less informative)

#### Step 2: Sort Features by Absolute Deviation

```python
# Sort all features descending by absolute deviation
stats_list.sort(key=lambda x: x['abs_deviation_pct'], reverse=True)
```

**Result:** Features ranked from most to least defining

#### Step 3: Select Top N Features

```python
# Only include top N most defining features in LLM prompt
top_features = stats[:max_stats_vars]  # Default: max_stats_vars = 10
```

**Configuration:**

- `prompt_max_stats_vars` (default: 10)
- Balances information richness vs. prompt conciseness

### Implementation

**Location:** `Cluster.compute_numeric_statistics()` in `pyhercules.py` (lines 917-1010)

**Key Code Snippet:**

```python
# Calculate deviation for each feature
for i in range(num_features):
    var_data = data[:, i]
    cluster_mean = np.mean(var_data)

    if global_means is not None:
        global_mean = global_means[i]
        deviation_pct = ((cluster_mean - global_mean) / abs(global_mean)) * 100
        abs_deviation_pct = abs(deviation_pct)

    stats_list.append({
        'var_name': variable_names[i],
        'mean': cluster_mean,
        'global_mean': global_mean,
        'deviation_pct': deviation_pct,
        'abs_deviation_pct': abs_deviation_pct,
        # ... other stats (median, min, max, std)
    })

# Sort by absolute deviation (descending) - most defining features first
stats_list.sort(key=lambda x: x['abs_deviation_pct'], reverse=True)
```

### Example Output

**Cluster Statistics (Top 3 of 25 features):**

| Feature        | Cluster Mean | Global Mean | Deviation | Abs. Deviation |
| -------------- | ------------ | ----------- | --------- | -------------- |
| MonthlyCharges | 85.50        | 64.76       | +32.04%   | **32.04%**     |
| TenureMonths   | 8.2          | 32.4        | -74.69%   | **74.69%**     |
| ContractMonth  | 0.95         | 0.45        | +111.11%  | **111.11%**    |
| ...            | ...          | ...         | ...       | ...            |

**LLM receives:** Only top 10 features with highest absolute deviations

**Benefit:**

- ✅ Focuses LLM attention on discriminative features
- ✅ Reduces prompt size (faster, cheaper inference)
- ✅ Improves description relevance by filtering noise

---

## 2. Contextualized LLM Prompting

### Objective

Generate cluster descriptions that use **domain-specific terminology** and align with **business objectives** by injecting context into LLM prompts.

### Mechanism 1: Industry Context

#### User Input (Dropdown)

**Available Options:**

- **Telco:** Telecommunications
- **Bank:** Banking & Finance
- **Retail:** Retail & E-commerce
- (Empty): No specific industry

**Location:** `pyhercules_app.py` - Industry dropdown in clustering configuration

#### Prompt Integration

When industry is selected:

```python
if industry_context:
    industry_names = {
        'telco': 'Telecommunications',
        'bank': 'Banking & Finance',
        'retail': 'Retail & E-commerce'
    }
    industry_display = industry_names.get(industry_context, industry_context)
    prompt += f"INDUSTRY CONTEXT: Use {industry_display} domain terminology and jargon in your responses.\n"
```

**Injected Prompt Snippet:**

```
INDUSTRY CONTEXT: Use Telecommunications domain terminology and jargon in your responses.
```

**Effect:**

- **Without context:** "Customers with high monthly charges and short tenure"
- **With Telco context:** "High-ARPU early churners with month-to-month contracts"
- **With Retail context:** "High-basket recent shoppers on trial subscriptions"

**Technology:** Prompt engineering via domain-specific instruction

---

### Mechanism 2: Topic Seed (Business Goal)

#### User Input (Text Field)

**Purpose:** Guide LLM to orient cluster descriptions toward specific business themes

**Examples:**

- "customer retention strategies"
- "product recommendation opportunities"
- "fraud detection patterns"
- "upsell potential"

**Location:** `pyhercules_app.py` - Topic seed input in clustering configuration

#### Prompt Integration

When topic seed is provided:

```python
if topic_seed:
    escaped_seed = topic_seed.replace('"', '\\"').replace("'", "\\'")
    prompt += f"TOPIC FOCUS: Orient towards '{escaped_seed}', if relevant.\n"
```

**Injected Prompt Snippet:**

```
TOPIC FOCUS: Orient towards 'customer retention strategies', if relevant.
```

**Effect:**

- **Descriptions emphasize retention-relevant characteristics**
- Suggestions focus on retention actions (e.g., "Target with loyalty offers")
- Language frames insights as retention opportunities

**Technology:** Goal-oriented prompt steering

---

### Complete LLM Prompt Structure

**Location:** `Hercules._build_llm_prompt()` in `pyhercules.py` (lines 2211-2390)

#### Full Template

```
Generate a concise 'title' (max 5-7 words) and 'description' (3-4 sentences)
for EACH of the Clusters below (Level 2).

[IF INDUSTRY SET]
INDUSTRY CONTEXT: Use {Industry} domain terminology and jargon in your responses.

[IF TOPIC SEED SET]
TOPIC FOCUS: Orient towards '{topic_seed}', if relevant.

RESPONSE FORMAT: Respond ONLY with a single, valid JSON object.
- Top-level keys MUST be the string representation of the 'Cluster ID' provided.
- Values MUST be JSON objects containing non-empty "title" and "description" string keys.

EXAMPLE (for IDs "cluster_5", "cluster_12"):
{
  "cluster_5": { "title": "High-Value Loyalists", "description": "Desc..." },
  "cluster_12": { "title": "At-Risk Churners", "description": "Desc..." }
}

IMPORTANT: Ensure the entire output is valid JSON. Do NOT include markdown fences.

--- Clusters Information ---

--- Cluster ID: cluster_5 (L2, 342 base items, BaseType: Numeric, InternalID: 5) ---
Representative Content/Samples (from L0 Descendants):
- (Orig. ID: customer_1023) "MonthlyCharges=89.50, TenureMonths=6, Contract=Month-to-month"
- (Orig. ID: customer_2451) "MonthlyCharges=95.20, TenureMonths=4, Contract=Month-to-month"
- (Orig. ID: customer_8912) "MonthlyCharges=82.10, TenureMonths=9, Contract=Month-to-month"

Most Defining Features (Top Deviations from Global Mean):
- TenureMonths: (Cluster: 6.8 vs Global: 32.4) | ↓ Deviation: 79.01%
- MonthlyCharges: (Cluster: 85.5 vs Global: 64.8) | ↑ Deviation: 32.04%
- ContractMonth-to-month: (Cluster: 1.0 vs Global: 0.45) | ↑ Deviation: 122.22%
- InternetServiceFiber: (Cluster: 0.92 vs Global: 0.44) | ↑ Deviation: 109.09%
- TechSupport_No: (Cluster: 0.88 vs Global: 0.49) | ↑ Deviation: 79.59%
- OnlineBackup_No: (Cluster: 0.85 vs Global: 0.45) | ↑ Deviation: 88.89%
- ... (19 more features with lower deviations)

--- End Cluster ID: cluster_5 ---

--- Cluster ID: cluster_12 (L2, 125 base items, ...) ---
[... similar structure for next cluster ...]

--- End Clusters Information ---

Generate the JSON output for the 2 Cluster ID(s): cluster_5, cluster_12
```

#### Prompt Components Breakdown

| Component                | Source                         | Purpose                                                    |
| ------------------------ | ------------------------------ | ---------------------------------------------------------- |
| **Task instruction**     | Fixed template                 | Define output format (JSON, title+description)             |
| **Industry context**     | User dropdown                  | Inject domain jargon                                       |
| **Topic seed**           | User text input                | Guide business focus                                       |
| **Format specification** | Fixed template                 | Ensure parseable JSON response                             |
| **Cluster ID**           | Cluster object                 | Unique identifier for mapping                              |
| **L0 samples**           | `get_representative_samples()` | Show raw data examples                                     |
| **Top features**         | `compute_numeric_statistics()` | **Feature-driven pre-analysis** (sorted by abs. deviation) |
| **Feature metadata**     | Variable metadata              | Add units, descriptions if available                       |
| **Deviation indicators** | Statistical calculation        | Show ↑/↓ direction and magnitude                           |

---

## Integration Flow

```
Raw Cluster Data
      ↓
[Compute Statistics]
  • Calculate cluster means
  • Calculate % deviation from global means
  • Calculate absolute deviations
      ↓
[Sort by Deviation]
  • Rank features by abs_deviation_pct (descending)
      ↓
[Filter Top N]
  • Select top max_stats_vars features (default: 10)
      ↓
[Build Prompt]
  • Add industry context (if set)
  • Add topic seed (if set)
  • Include L0 samples
  • Include top features with deviations
      ↓
[LLM Generation]
  • Generate title & description
  • Apply domain terminology (from industry)
  • Align with business goal (from topic seed)
      ↓
Actionable Cluster Labels
```

---

## Configuration Parameters

### Feature Selection

**Parameter:** `prompt_max_stats_vars`  
**Default:** 10  
**Type:** Integer (1-50 recommended)  
**Effect:** Number of top features included in LLM prompt

```python
hercules = Hercules(
    prompt_max_stats_vars=10  # Show only top 10 most defining features
)
```

### Industry Context

**Parameter:** `industry_context`  
**Type:** String (`'telco'`, `'bank'`, `'retail'`, or custom)  
**Effect:** Guides LLM to use industry-specific terminology

```python
hercules = Hercules(
    industry_context='telco'  # Inject telecommunications jargon
)
```

### Topic Seed

**Parameter:** `topic_seed` (in `cluster()` method)  
**Type:** String  
**Effect:** Orients descriptions toward business theme

```python
top_clusters = hercules.cluster(
    data,
    topic_seed="customer retention strategies"  # Focus on retention
)
```

---

## Technical Implementation

### Key Functions

| Function                       | Location                   | Purpose                                  |
| ------------------------------ | -------------------------- | ---------------------------------------- |
| `compute_numeric_statistics()` | `pyhercules.py` L917-1010  | Calculate deviations, sort by abs. value |
| `get_data_for_prompt()`        | `pyhercules.py` L1015-1090 | Prepare cluster data for LLM             |
| `_build_llm_prompt()`          | `pyhercules.py` L2211-2390 | Construct complete prompt with context   |

### Data Flow

```python
# 1. Compute statistics with global means (for deviation)
stats = cluster.compute_numeric_statistics(
    variable_names=['Age', 'Income', 'Tenure', ...],
    global_means=hercules._global_numeric_means_  # Global dataset means
)
# → Returns dict sorted by abs_deviation_pct (descending)

# 2. Prepare prompt data (includes top N stats)
prompt_data = cluster.get_data_for_prompt(
    max_stats_vars=10,  # Filter to top 10 features
    # ... other params
)
# → Includes only highest-deviation features

# 3. Build full prompt with context
prompt = hercules._build_llm_prompt(
    [prompt_data_cluster1, prompt_data_cluster2, ...]
)
# → Injects industry_context and topic_seed
# → Formats top features with deviation indicators

# 4. Send to LLM
response = llm_function(prompt)
# → LLM generates descriptions using context + top features
```

---

## Example: End-to-End

### Input Configuration

```python
hercules = Hercules(
    level_cluster_counts=[5],
    industry_context='telco',
    prompt_max_stats_vars=5  # Show only top 5 features
)

top_clusters = hercules.cluster(
    telco_customer_data,
    topic_seed='churn prediction'
)
```

### Feature-Driven Pre-Analysis Output

**Cluster 3 - All Features (25 total):**

| Rank | Feature                | Abs. Deviation |
| ---- | ---------------------- | -------------- |
| 1    | TenureMonths           | 79.01%         |
| 2    | ContractMonth-to-month | 122.22%        |
| 3    | MonthlyCharges         | 32.04%         |
| 4    | InternetServiceFiber   | 109.09%        |
| 5    | TechSupport_No         | 79.59%         |
| 6    | OnlineBackup_No        | 45.21%         |
| ...  | ...                    | ...            |
| 25   | Dependents_Yes         | 2.34%          |

**Filtered to Top 5** (sent to LLM)

### Contextualized Prompt (Excerpt)

```
INDUSTRY CONTEXT: Use Telecommunications domain terminology and jargon in your responses.
TOPIC FOCUS: Orient towards 'churn prediction', if relevant.

--- Cluster ID: cluster_3 ---
Most Defining Features (Top Deviations from Global Mean):
- TenureMonths: (Cluster: 6.8 vs Global: 32.4) | ↓ Deviation: 79.01%
- ContractMonth-to-month: (Cluster: 1.0 vs Global: 0.45) | ↑ Deviation: 122.22%
- MonthlyCharges: (Cluster: 85.5 vs Global: 64.8) | ↑ Deviation: 32.04%
- InternetServiceFiber: (Cluster: 0.92 vs Global: 0.44) | ↑ Deviation: 109.09%
- TechSupport_No: (Cluster: 0.88 vs Global: 0.49) | ↑ Deviation: 79.59%
```

### LLM Output

```json
{
  "cluster_3": {
    "title": "High-ARPU Early-Tenure Churn Risk",
    "description": "Fiber subscribers with premium monthly charges (↑32%) but critically low tenure (↓79%). Nearly all on month-to-month contracts without tech support add-ons. Strong churn indicators suggest immediate retention intervention needed—consider contract upgrade incentives and proactive support outreach."
  }
}
```

**Note:**

- ✅ Uses telco jargon: "ARPU", "Fiber subscribers", "contract upgrade"
- ✅ Oriented to churn prediction: "churn indicators", "retention intervention"
- ✅ Focuses on top 5 features (ignores Dependents_Yes at 2.34% deviation)
- ✅ Provides actionable suggestion aligned with topic seed

---

## Benefits

### Feature-Driven Pre-Analysis

| Benefit               | Impact                                        |
| --------------------- | --------------------------------------------- |
| **Statistical rigor** | Prioritizes objectively defining features     |
| **Noise reduction**   | Filters out low-signal features               |
| **Prompt efficiency** | Reduces token count, faster/cheaper LLM calls |
| **Interpretability**  | Highlights what makes clusters unique         |

### Contextualized Prompting

| Benefit              | Impact                                          |
| -------------------- | ----------------------------------------------- |
| **Domain relevance** | Descriptions use familiar business language     |
| **Actionability**    | Suggestions align with business goals           |
| **Flexibility**      | Same data → different insights based on context |
| **User control**     | Business users guide AI output without coding   |

---

## Summary

**Actionable Summarization Layer** = Statistical Feature Filtering + Contextual LLM Guidance

- **Feature-Driven Pre-Analysis:** Absolute z-score deviation ranking ensures LLM sees only the most defining characteristics
- **Contextualized Prompting:** Industry context + topic seed inject domain knowledge and business objectives

**Result:** AI-generated cluster descriptions that are statistically grounded, domain-specific, and aligned with business goals.

**Technologies:** NumPy (statistical computation), prompt engineering (LLM steering), UI-driven context injection (Dash)
