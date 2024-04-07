import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# paths
train_path = './data/data_train.csv'
test_path = './data/data_test.csv'
predictions_path = './data/s4000577_predictions_decision_tree.csv'

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Separate features and target from training data
x_train = train_df.drop(columns=['Id', 'Status'])
y_train = train_df['Status']

# Preprocessing:
# Impute missing values if there are any.
# We will use median for imputation because it's robust to outliers.
# Then scale the features since we're using a tree-based model which might not need scaling but it maintains consistency with previous models.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), x_train.columns)
    ])

# Define the model pipeline
pipeline = make_pipeline(preprocessor, DecisionTreeClassifier(random_state=42))

# Split the training data into a smaller training subset and a validation subset
x_train_sub, x_val, y_train_sub, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Fit the model on the reduced training data
pipeline.fit(x_train_sub, y_train_sub)

# Predict on the validation set
y_val_pred = pipeline.predict(x_val)

# Calculate accuracy and other performance metrics on the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
val_classification_report = classification_report(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)
print("Validation Classification Report:\n", val_classification_report)

# Evaluate the model using cross-validation on the full training data
cv_scores = cross_val_score(pipeline, x_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores.mean())

# Prediction on test set (without the 'Status' column since it's what we're supposed to predict)
x_test = test_df.drop(columns=['Id', 'Status'])
predicted_status = pipeline.predict(x_test)

# Adding predictions to the test dataframe
test_df['Status'] = predicted_status

# Save the predictions to a CSV file
test_df[['Id', 'Status']].to_csv(predictions_path, index=False)