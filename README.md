# House Price Predictor

A regression project that estimates house prices from structural and amenity features. The notebook moves from exploratory analysis and categorical encoding to linear and tree-based model comparison, hyperparameter tuning, and artifact export.

## Workflow

- Explore price, area, room counts, amenities, furnishing status, and outliers.
- Convert yes/no attributes to binary values and one-hot encode furnishing status.
- Split the data into training and test sets.
- Standardize features for linear regression.
- Compare linear regression with random-forest regression.
- Tune the random forest with cross-validated grid search scored by mean absolute error.
- Report MAE, RMSE, and R-squared.
- Export the fitted linear model and standard scaler.

## Notebook walkthrough

1. Inspect the housing table and explore price by area, room count, parking, furnishings, and amenities. Use histograms, scatter plots, correlations, group means, and box plots to examine patterns and outliers.
2. Convert yes/no amenity fields to binary values and one-hot encode furnishing status, dropping one reference category.
3. Split predictors from `price`, create a reproducible train/test partition, and fit a standard scaler on the training features for linear regression.
4. Fit linear regression, inspect its coefficients and example predictions, and compute mean absolute error, root mean squared error, and R-squared.
5. Fit a 100-tree random forest on the unscaled feature table, then tune a forest with grid search scored by negative mean absolute error. Compare the tuned forest with the linear baseline on the held-out set.
6. Export the **linear regression** model and its scaler. The tuned forest is evaluated in the notebook but is not the model saved as `house_price_model.pkl`; inference also needs the same encoding and column order.

## Dataset

`Housing.csv` contains 545 homes. Predictors include area, bedrooms, bathrooms, stories, road access, guest room, basement, hot-water heating, air conditioning, parking, preferred-area status, and furnishing status. The target is `price`.

## Repository contents

| File | Purpose |
| --- | --- |
| `House-Price-Predictor.ipynb` | Exploration, preprocessing, regression, tuning, and evaluation |
| `Housing.csv` | Source housing dataset |
| `house_price_model.pkl` | Exported linear-regression model |
| `scaler.pkl` | Exported feature scaler |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter lab House-Price-Predictor.ipynb
```

The notebook displays currency using the rupee symbol, but the dataset should be consulted to confirm the intended unit before real-world interpretation.
