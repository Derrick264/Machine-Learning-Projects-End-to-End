# DATA_PREPARATION.md

## Data Preparation: Exploratory Data Analysis, Cleaning, and Feature Engineering

**This document provides a step-by-step record of the entire data preparation pipeline for the movie success prediction project. Every major step is justified with reasoning to support both reproducibility and transparency.**

---

### 1. Exploratory Data Analysis (EDA)

#### **A. Data Overview**
- Inspected the dataset with `.head()`, `.info()`, and `.describe()` to check structure, feature types, and sample content.
- **Why:** Understand basic characteristics, spot immediate errors or issues, and organize next steps.

#### **B. Missing Value Detection & Visualization**
- Counted nulls per column and visualized missingness using `missingno`.
- **Why:** Gauge which columns needed imputation, how severe missingness was, and whether to drop or retain certain features.

#### **C. Duplicate Detection**
- Removed all duplicate rows.
- **Why:** Duplicates can skew learning, bias results, and artificially increase data set size.

#### **D. Feature Distribution & Outlier Analysis**
- Plotted histograms and boxplots for numeric features (budget, gross, Facebook likes, etc.).
- **Why:** Detected right-skewed distributions and outliers, informing the need for log transformation and/or capping.

#### **E. Correlation & Feature Relationship Analysis**
- Used correlation matrices and visual heatmaps to identify strong predictors and multicollinearity.
- **Why:** Pinpointed important predictors for success, and found candidate columns for combination or redundancy removal.

---

### 2. Data Cleaning

#### **A. Handling Missing Values**  
- Numeric columns: Filled with median values—robust to outliers.
- Categorical/string columns: Filled with `'Unknown'` to retain rows.
- `title_year`: Imputed with its mode, and cast to integer type.
- **Why:** ML models require complete data; choices reduce error propagation and bias.

#### **B. Data Type Consistency**
- Ensured all features were properly typed (numeric for modeling, categorical where applicable).
- **Why:** Guarantees function compatibility throughout the pipeline.

#### **C. Dropping Unnecessary Columns**
- Dropped columns unlikely to contribute predictive value, such as `movie_title`, `movie_imdb_link`, and `genres_list` (after encoding).
- `plot_keywords` dropped unless NLP-based features were added.
- **Why:** Reduces noise, memory usage, and potential information leakage.

---

### 3. Feature Engineering

#### **A. Grouping Rare Categories**
- For `director_name`, `actor_1_name`, and `actor_2_name`, grouped all names appearing <5 times as `'Other'`.
- **Why:** Controls high cardinality, reduces noise and risk of overfitting, while retaining main predictive contributors.

#### **B. Categorical Encoding**
- Applied label encoding to main categorical actor/director fields and other category columns.
- Used multi-label binarization for genres.
- **Why:** ML models require numeric input, and this encoding approach preserves information while controlling dimensionality.

#### **C. Log Transformation**
- Applied `np.log1p()` to skewed columns: `budget`, `gross`, Facebook likes, etc.
- **Why:** Reduces the impact of long-tailed outliers and normalizes distributions for better model learning.

#### **D. Derived Features**
- `star_power`: Sum of lead actors' and director's Facebook likes.
- `movie_age`: Calculated as `current year - title_year`.
- `roi`: Computed as `gross / budget`.
- **Why:** These composites embed domain knowledge, allowing deeper capture of success drivers.

#### **E. Target Variable Creation**
- Created `success_category`: 
    - **Flop:** ROI < 1
    - **Average:** 1 ≤ ROI < 2
    - **Hit:** ROI ≥ 2
- **Why:** This aligns the classification problem with business logic and industry standards.

---

### 4. Handling Remaining Null Values

- **Systematically checked all columns for remaining nulls.**
- For numeric columns: Filled with median values.
- For encoded categorical columns: Filled with -1 if label-encoded, or 'Unknown' for string-based columns.
- For genre binary columns: Filled with 0 (absence of genre).
- For derived or composite columns (e.g., `star_power`): Filled with median or 0.
- Rows missing target variables (`gross`, `roi`) were **dropped**.
- **Why:** Complete, clean data ensures no runtime errors and higher modeling reliability.

---

### 5. Final Dataset Export

- **Saved the cleaned, fully processed dataframe** to `data/final_featured_movie_data.csv`.
- **Why:** Ensures reproducibility, version control, and efficient reloading for modeling.

---

## Summary Table

| Step               | What was done                                     | Why                   |
|--------------------|---------------------------------------------------|-----------------------|
| EDA                | Visualization, missing checks, duplicates, stats  | Understanding         |
| Imputation         | Medians (numeric), 'Unknown' (categorical)        | Completeness, robustness |
| Rare Category Grp  | Grouped directors/actors <5 movies as 'Other'     | Reduce cardinality    |
| Categorical Encode | Label encoding, MultiLabelBinarizer for genres    | Model compatibility   |
| Log Transform      | Log1p for skewed numeric/binary columns           | Normalize/scale       |
| Derived Features   | star_power, movie_age, roi                        | Domain knowledge      |
| Target Creation    | ROI-based success_category                        | Business relevance    |
| Drop/Fill Nulls    | Removed or imputed all missing values             | Clean for ML          |
| Export             | Saved as preprocessed CSV                         | Reproducibility       |

---

**This document should be kept updated as further data transformations or modeling steps are taken. All decisions are based on best data science practices and tailored for this movie success prediction project.**
