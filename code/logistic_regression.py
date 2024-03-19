import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# paths
train_path = './data/data_train.csv'
test_path = './data/data_test.csv'
predictions_path = './data/s4000577_predictions.csv'


# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


# Separate features and target from training data
x_train = train_df.drop(columns=['Id', 'Status'])
y_train = train_df['Status']


# Preprocessing: 
# Impute missing values if there are any. 
# We will use median for imputation because it's robust to outliers.
# Then scale the features since we'll use logistic regression.
preprocessor = ColumnTransformer(
  transformers=[
    ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), x_train.columns)
  ])


# Define the model pipeline
pipeline = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))

# Fit the model
pipeline.fit(x_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(pipeline, x_train, y_train, cv=5)

# Prediction on test set (without the 'Status' column since it's what we're supposed to predict)
x_test = test_df.drop(columns=['Id', 'Status'])
predicted_status = pipeline.predict(x_test)

# Adding predictions to the test dataframe
test_df['Status'] = predicted_status

# Save the predictions to a CSV file
test_df[['Id', 'Status']].to_csv(predictions_path, index=False)