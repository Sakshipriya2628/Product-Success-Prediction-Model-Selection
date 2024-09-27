import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.le_dict = {}
        
    def load(self, file_path):
        return pd.read_csv(file_path)
    
    def preprocess(self, df, is_training=True):
        for col in ['category', 'main_promotion', 'color']:
            if is_training:
                self.le_dict[col] = LabelEncoder()
                df[col] = self.le_dict[col].fit_transform(df[col])
            else:
                # Handle unseen categories
                le = self.le_dict[col]
                df[col] = df[col].map(lambda s: s if s in le.classes_ else 'Unknown')
                le.classes_ = np.append(le.classes_, 'Unknown')
                df[col] = le.transform(df[col])
        
        if is_training:
            X = df.drop('success_indicator', axis=1)
            y = df['success_indicator']
            return X, y
        else:
            return df
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        return X_test, y_test
    
    def test(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def predict(self, X):
        return self.model.predict(X)

if __name__ == "__main__":
    rf_model = RandomForestModel()
    
    # Load and preprocess the historic data
    df = rf_model.load('historic.csv')
    X, y = rf_model.preprocess(df, is_training=True)
    
    # Train the model and get the test set
    X_test, y_test = rf_model.train(X, y)
    
    # Test the model and print evaluation metrics
    rf_model.test(X_test, y_test)
    
    # Predict on new data
    new_data = rf_model.load('prediction_input.csv')
    new_X = rf_model.preprocess(new_data, is_training=False)
    predictions = rf_model.predict(new_X)
    print("\nPredictions for new data:", predictions[:10])  # Show first 10 predictions
    
    # Print unique values in each categorical column
    for col in ['category', 'main_promotion', 'color']:
        print(f"\nUnique values in {col}:")
        print(new_data[col].unique())