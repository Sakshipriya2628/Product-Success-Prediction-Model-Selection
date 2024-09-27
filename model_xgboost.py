import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.le_dict = {}
        self.target_le = LabelEncoder()
        
    def load(self, file_path):
        return pd.read_csv(file_path)
    
    def preprocess(self, df, is_training=True):
        for col in ['category', 'main_promotion', 'color']:
            if is_training:
                self.le_dict[col] = LabelEncoder()
                df[col] = self.le_dict[col].fit_transform(df[col])
            else:
                le = self.le_dict[col]
                df[col] = df[col].map(lambda s: s if s in le.classes_ else 'Unknown')
                if 'Unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'Unknown')
                df[col] = le.transform(df[col])
        
        if is_training:
            X = df.drop('success_indicator', axis=1)
            y = self.target_le.fit_transform(df['success_indicator'])
            print("Unique values in encoded y:", np.unique(y))  # Debugging print
            return X, y
        else:
            return df
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Shape of X_train:", X_train.shape)  # Debugging print
        print("Shape of y_train:", y_train.shape)  # Debugging print
        print("Unique values in y_train:", np.unique(y_train))  # Debugging print
        
        # Define the parameter grid for GridSearchCV
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Initialize XGBoost classifier
        xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, 
                                   cv=3, n_jobs=-1, verbose=2)
        
        try:
            grid_search.fit(X_train, y_train)
        except Exception as e:
            print("Error during grid search:", str(e))
            print("y_train values:", y_train)
            raise
        
        # Get the best model
        self.model = grid_search.best_estimator_
        
        print("Best parameters found: ", grid_search.best_params_)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
        
        return X_test, y_test
    
    def test(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.target_le.classes_))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def predict(self, X):
        return self.target_le.inverse_transform(self.model.predict(X))

if __name__ == "__main__":
    xgb_model = XGBoostModel()
    
    # Load and preprocess the historic data
    df = xgb_model.load('historic.csv')
    X, y = xgb_model.preprocess(df, is_training=True)
    
    # Train the model and get the test set
    X_test, y_test = xgb_model.train(X, y)
    
    # Test the model and print evaluation metrics
    xgb_model.test(X_test, y_test)
    
    # Predict on new data
    new_data = xgb_model.load('prediction_input.csv')
    new_X = xgb_model.preprocess(new_data, is_training=False)
    predictions = xgb_model.predict(new_X)
    print("\nPredictions for new data:", predictions[:10])  # Show first 10 predictions
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))