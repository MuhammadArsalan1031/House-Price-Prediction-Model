import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# 1. Dataset Generation
def generate_data(n_rows=500):
    """
    Generates a synthetic dataset for house price prediction.
    Features: area, bedrooms, bathrooms, location.
    Target: price.
    """
    np.random.seed(42)
    area = np.random.randint(600, 5000, n_rows)
    bedrooms = np.random.randint(1, 6, n_rows)
    bathrooms = np.random.randint(1, 5, n_rows)
    location = np.random.randint(1, 6, n_rows)
    
    # Updated Price formula for more 'realistic' results:
    # Base: 50,000
    # Area: $150 / sqft
    # BR: $25,000 each
    # Bath: $15,000 each
    # Loc: $40,000 per rating level
    noise = np.random.randint(-10000, 10000, n_rows)
    price = 50000 + (area * 150) + (bedrooms * 25000) + (bathrooms * 15000) + (location * 40000) + noise
    
    df = pd.DataFrame({
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'location': location,
        'price': price
    })
    return df

def train_and_save_model():
    print("Step 1: Generating synthetic dataset (500 rows)...")
    df = generate_data(500)
    df.to_csv('data.csv', index=False)
    print("Dataset saved as 'data.csv'.")

    # Step 2: Data Processing
    print("Step 2: Splitting data into train/test sets...")
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Model Training
    print("Step 3: Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: Evaluation
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    print(f"Model Training Complete. R2 Score (Accuracy): {accuracy:.4f}")

    # Step 5: Verification
    print("\nTest Predictions:")
    test_samples = pd.DataFrame([
        [1500, 3, 2, 3], # Typical house
        [2500, 4, 3, 4], # Premium house
        [800, 1, 1, 1]   # Small house
    ], columns=['area', 'bedrooms', 'bathrooms', 'location'])
    predictions = model.predict(test_samples)
    print(f"1500sqft, 3BR, 2Bath, Loc 3 -> ${predictions[0]:,.2f}")
    print(f"2500sqft, 4BR, 3Bath, Loc 4 -> ${predictions[1]:,.2f}")
    print(f"800sqft, 1BR, 1Bath, Loc 1 -> ${predictions[2]:,.2f}")

    # Step 6: Model Saving
    joblib.dump(model, 'model.pkl')
    print("\nModel saved as 'model.pkl'.")

if __name__ == "__main__":
    train_and_save_model()
