import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import joblib
import warnings
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.regularizers import l1_l2

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message=".*")

# Define numerical and categorical features for prediction
numerical_features = ['sched_dep_time', 'distance', 'month', 'day', 'hour', 'minute']
categorical_features = ['carrier', 'origin', 'dest']

# Function to preprocess the data
def preprocess_data(df):
    # Dropping rows with missing target (dep_delay)
    df = df.dropna(subset=['dep_delay'])
    # Creating a binary feature for delay: 1 if dep_delay >= 15 else 0
    df['is_delayed'] = (df['dep_delay'] >= 15).astype(int)
    # Splitting data into features and target
    X = df[numerical_features + categorical_features]
    y = df['is_delayed']
    return X, y

# Preprocessing pipeline with KNN Imputer
def create_preprocessing_pipeline():
    # Preprocessing for numerical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),  # KNN Imputer
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Bundle preprocessing for numeric and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Custom callback to compute F1 score and save the best model
class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data
        self.best_f1 = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        val_data, val_labels = self.validation_data
        val_pred = (np.asarray(self.model.predict(val_data))).argmax(axis=1)
        val_true = val_labels.argmax(axis=1)
        current_f1 = f1_score(val_true, val_pred)

        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_weights = self.model.get_weights()

        print(f"Epoch {epoch + 1}: Validation F1 Score: {current_f1:.4f}, Best F1 Score: {self.best_f1:.4f}")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"Restored best model with F1 Score: {self.best_f1:.4f}")

# Load and process the data
file_path = 'flight_data.csv'  # Replace with your local file path
flight_data = pd.read_csv(file_path)
X, y = preprocess_data(flight_data)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and apply the preprocessing pipeline
preprocessing_pipeline = create_preprocessing_pipeline()
X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
X_test_transformed = preprocessing_pipeline.transform(X_test)

# Outlier detection using IsolationForest
isolation_forest = IsolationForest(contamination=0.01, random_state=42)
outliers = isolation_forest.fit_predict(X_train_transformed)

# Filter out the outliers
X_train_filtered = X_train_transformed[outliers == 1]
y_train_filtered = y_train[outliers == 1]

# SMOTE for oversampling the minority class
smote = SMOTE(random_state=42)

# Apply SMOTE to the filtered training data
X_resampled, y_resampled = smote.fit_resample(X_train_filtered, y_train_filtered)

# Convert labels to categorical format for neural network
y_resampled_categorical = to_categorical(y_resampled, num_classes=2)
y_test_categorical = to_categorical(y_test, num_classes=2)

# Neural Network Model with mixed activation functions, batch normalization, and L1/L2 regularization
model = Sequential()
model.add(Dense(128, input_dim=X_resampled.shape[1], activation='tanh', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # ReLU activation
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='tanh', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # tanh activation
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # ReLU activation
model.add(BatchNormalization())
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss=metrics.BinaryCrossentropy(), metrics=['accuracy',metrics.F1Score(average="weighted")])

# F1 Score Callback
f1_score_callback = F1ScoreCallback(validation_data=(X_test_transformed, y_test_categorical))

# Train the model with F1 score monitoring
model.fit(X_resampled, y_resampled_categorical, epochs=100, batch_size=64,
          validation_data=(X_test_transformed, y_test_categorical),
          callbacks=[f1_score_callback])

# Evaluate the model on the test set
y_pred_proba = model.predict(X_test_transformed)
y_pred = np.argmax(y_pred_proba, axis=1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Save the trained model
model.save('best_nn_model_mixed_activations_bn_l1_l2.h5')

# Save evaluation metrics to a file
with open('evaluation_metrics_nn_mixed_activations_bn_l1_l2.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"ROC-AUC: {roc_auc}\n")
    f.write(f"Confusion Matrix:\n{cm}\n")

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")
print(f"Confusion Matrix:\n{cm}")
