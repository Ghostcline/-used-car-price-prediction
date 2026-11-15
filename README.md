# Used Car Price Prediction

A machine learning project that predicts used car prices based on vehicle information such as model year, mileage, brand, model, and other features.

I built this project after completing the first course of Andrew Ng's Machine Learning Specialization. My goal was to move beyond understanding regression in theory and learn how to build, evaluate, and improve a model using real data.

## Dataset

The project uses the [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset) from Kaggle, containing **4,009 vehicle listings**.

The dataset includes features such as:

- Brand and model
- Model year
- Mileage
- Fuel type and transmission
- Accident history
- Clean title status

The target variable is the vehicle's listed price in USD.

## Approach

### 1. Data Exploration and Cleaning

I started by inspecting the dataset, checking missing values, converting price and mileage into numerical values, and exploring relationships between features and prices.

One interesting observation was that the price distribution was heavily right-skewed, with a few extremely expensive vehicles.

### 2. Linear Regression from Scratch

Before using Scikit-learn's regression models, I implemented linear regression myself using Python and NumPy.

This included:

- Prediction function
- Cost function
- Gradient calculations
- Gradient descent
- Training and model evaluation

I started with just two features: model year and mileage.

This helped me understand how weights and bias are updated during training.

### 3. Scikit-learn Pipeline

I then built a more complete model using eight input features.

I used Scikit-learn Pipelines to handle missing values, numerical scaling, and categorical encoding.

I compared standard Linear Regression with Ridge Regression and tested different regularization strengths.

I also experimented with grouping rare categorical values, but this reduced validation performance, so I kept the original encoding approach.

## Model Results

The selected model was **Ridge Regression with alpha = 0.1**.

| Metric | Validation | Test |
|---|---:|---:|
| MAE | $11,972.19 | $20,930.82 |
| RMSE | $24,042.33 | $137,580.70 |
| R² | 0.7001 | 0.0739 |

The model performed considerably better on validation data than on the test set.

Further error analysis showed that the three largest prediction errors accounted for **95.11% of the total squared error on the test set**.

These errors mainly involved unusually expensive vehicles that were difficult to predict accurately.

The test set also had some overlap with data inspected during earlier model development, so its results should not be treated as a completely independent estimate of generalization.

## What I Learned

This project helped me understand that training a model is only one part of machine learning.

Some of my main takeaways were:

- Data quality and feature selection matter as much as the algorithm.
- A lower training error does not necessarily mean better predictions on new data.
- More features and stronger regularization do not always improve performance.
- Evaluating a model with multiple metrics is important, especially when the dataset contains extreme values.
- Understanding where a model fails can be more useful than looking at one overall score.

## Running the Project

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies from `requirements.txt`.
4. Download the dataset from Kaggle and place `used_cars.csv` inside the `data/` directory.
5. Open `notebooks/01_used_car_prediction.ipynb` in VS Code or Jupyter.
6. Select the Python environment and run the notebook.

The notebook uses relative paths that assume the working directory is `notebooks/`. Adjust the paths if your environment uses the repository root instead.

The trained pipeline can be saved to `models/used_car_model.joblib`.

## Limitations

This is a learning project, not a production-ready car valuation system.

The dataset is relatively small, some listings contain unusually high prices, and the model struggles with rare or high-value vehicles.

The predictions should not be treated as reliable real-world vehicle valuations.

## Tools Used

Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook, and Joblib.