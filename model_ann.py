import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

class ANNModel:
    def __init__(self):
        self.model = None
        self.le_dict = {}
        self.scaler = StandardScaler()
        self.target_le = LabelEncoder()
        
    def load(self, file_path):
        return pd.read_csv(file_path)
    
    def preprocess(self, df, is_training=True):
        # Encode categorical variables
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
            # Encode the target variable
            y = self.target_le.fit_transform(df['success_indicator'])
            X = df.drop('success_indicator', axis=1)
        else:
            X = df
            y = None
        
        # Scale features
        X = pd.DataFrame(self.scaler.fit_transform(X) if is_training else self.scaler.transform(X),
                         columns=X.columns)
        
        return X, y
    
    def build_model(self, input_shape):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.build_model(X_train.shape[1])
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
        return X_test, y_test
    
    def test(self, X_test, y_test):
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.target_le.classes_))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def predict(self, X):
        return self.target_le.inverse_transform((self.model.predict(X) > 0.5).astype(int))

if __name__ == "__main__":
    ann_model = ANNModel()
    
    # Load and preprocess the historic data
    df = ann_model.load('historic.csv')
    X, y = ann_model.preprocess(df, is_training=True)
    
    # Train the model and get the test set
    X_test, y_test = ann_model.train(X, y)
    
    # Test the model and print evaluation metrics
    ann_model.test(X_test, y_test)
    
    # Predict on new data
    new_data = ann_model.load('prediction_input.csv')
    new_X, _ = ann_model.preprocess(new_data, is_training=False)
    predictions = ann_model.predict(new_X)
    print("\nPredictions for new data:", predictions[:10])  # Show first 10 predictions