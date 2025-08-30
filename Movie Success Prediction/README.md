# Movie Success Prediction 

## Project Overview
This project aims to predict the commercial success category (Hit, Average, or Flop) of movies based on their metadata, social media popularity indicators, and other derived features using machine learning. By leveraging publicly available movie datasets and social media engagement data, the objective is to identify key factors influencing box office outcomes and develop predictive models to classify movies according to their success level.

## Dataset Description
The dataset comprises approximately 5,000 movies with 28 original columns capturing a mixture of:

- **Numerical features:** budget, gross revenue, movie duration, number of critic reviews, Facebook likes of directors and actors, user votes, and more.
- **Categorical features:** director names, actor names, movie color, language, content rating, genres, and keywords.
- **Textual fields:** movie titles and IMDb links.
- **Target variable:** success category derived from the return on investment (ROI), computed as gross revenue divided by budget.

The data spans multiple decades and regions, providing a rich variety of movies and social contexts.

## Data Preparation
A comprehensive data preparation pipeline was developed encompassing the following key stages:

### 1. Exploratory Data Analysis (EDA)
- Conducted initial inspection of data types, distributions, and missing values.
- Visualized numeric feature distributions (e.g., budget, gross) revealing typical right-skewed patterns.
- Identified and removed duplicate records to prevent bias.
- Examined correlations between features and the target variable, highlighting budget, star power, and engagement metrics as primary predictors.

### 2. Data Cleaning
- Imputed missing values:
  - Numeric columns were filled using medians to handle outliers effectively.
  - Categorical and text columns were assigned a placeholder `'Unknown'` to preserve row integrity.
- Corrected data types for features such as release year (`title_year`), ensuring consistent numeric typing.
- Removed non-informative columns like movie titles and IMDb links to reduce noise.

### 3. Feature Engineering
- Grouped rare categories in high-cardinality columns (director and actor names) by aggregating those with fewer than five movies under an `"Other"` label before encoding, which mitigates sparsity and overfitting risks.
- Converted categorical variables to numeric format with label encoding for compatibility with machine learning models.
- Applied multi-label binarization to the `genres` column, resulting in distinct binary columns per genre, allowing nuanced genre inclusion.
- Created log-transformed versions of skewed features (`budget`, `gross`, Facebook likes) to normalize distributions and stabilize model training.
- Constructed composite features including:
  - `star_power`: Combined Facebook likes of lead actors and director to quantify celebrity influence.
  - `movie_age`: Age of the movie computed relative to a reference year to capture temporal effects.
  - `roi`: Return on investment as gross divided by budget, foundational for defining success categories.

### 4. Target Variable Definition
- Defined the `success_category` target variable based on ROI thresholds:
  - **Flop:** ROI < 1 (gross does not cover budget).
  - **Average:** 1 ≤ ROI < 2 (modest profitability).
  - **Hit:** ROI ≥ 2 (successful, at least double budget).
- This business-driven categorization enables clear classification objectives aligned with industry understanding.

### 5. Final Dataset Preparation
- After encoding and transformations, ensured all features are numeric and contain no missing values.
- Verified absence of outliers and multicollinearity issues.
- Saved finalized preprocessed data for subsequent modeling steps.

> For a detailed, step-by-step walkthrough of all data preparation processes, including code snippets and exploratory analyses, please see the [`DATA_PREPARATION.md`](DATA_PREPARATION.md) document.

## Modeling Approach
- Employed classical supervised machine learning classifiers to predict movie success category.
- Initial models included Random Forest and Gradient Boosting classifiers chosen for their robustness and ability to handle mixed data types.
- Utilized stratified train-test splits to maintain balanced class distribution.
- Performed hyperparameter tuning using grid search cross-validation to optimize model parameters such as tree depth, number of estimators, and minimum samples per leaf.
- Evaluated models primarily using classification metrics: accuracy, precision, recall, F1-score, and ROC-AUC (macro average).
- Interpreted feature importance to understand key predictors influencing predictions.

## Results and Evaluation
- Both Random Forest and Gradient Boosting models achieved strong predictive performance, with accuracy typically above 75% on held-out test data.
- Confusion matrices revealed the models were particularly effective at identifying “Hit” and “Flop” classes, with minor confusion around the “Average” category.
- ROC-AUC scores averaged around 0.80 across classes, indicating good discriminatory power.
- Feature importance analyses highlighted that budget (and log budget), star power, number of user votes, and movie age were the most influential features.
- The results demonstrate the feasibility of predicting movie success from metadata and social media engagement indicators with reasonable accuracy.

## Future Work
- Integrate natural language processing on plot keywords and plot summaries to capture semantic movie elements.
- Explore deep learning models and embeddings for improved feature representation.
- Experiment with ensemble methods combining multiple classifiers.
- Implement model calibration and uncertainty estimation for more reliable predictions.
- Develop a production-grade deployment pipeline with continuous integration and automated updates.
- Incorporate user feedback in the deployed app to iteratively improve predictions.

## How to Run

### Requirements
- Python 3.8 or higher
- Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib, missingno, joblib, streamlit

### Execution
1. Prepare your Python environment using the provided `requirements.txt` file.
2. Follow the notebook sequence in the `notebooks/` directory:
   - `01_data_acquisition_cleaning_and_eda.ipynb`
   - `04_modeling_and_evaluation.ipynb`
3. Load preprocessed data from `data/final_cleaned_featured_movie_data.csv` to run modeling.
4. Use the Streamlit app located at `scripts/app.py` for interactive predictions:

