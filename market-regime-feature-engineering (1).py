import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

def engineer_market_regime_features(ohlcv_data):
    """
    Generate features for market regime classification
    
    Parameters:
    -----------
    ohlcv_data : pandas.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume
    
    Returns:
    --------
    features : numpy.ndarray
        Engineered features for classification
    """
    df = ohlcv_data.copy()
    
    # Calculate price-based features
    df['price_change'] = df['Close'].pct_change()
    df['price_momentum'] = df['Close'].rolling(window=5).mean().pct_change()
    df['price_volatility'] = df['Close'].rolling(window=5).std()
    
    # Calculate volume-based features
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_momentum'] = df['Volume'].rolling(window=5).mean().pct_change()
    
    # Identify consecutive higher/lower highs and lows
    df['consec_higher_highs'] = (df['High'] > df['High'].shift(1)).rolling(window=3).sum()
    df['consec_lower_lows'] = (df['Low'] < df['Low'].shift(1)).rolling(window=3).sum()
    
    # Drop NaN rows
    df.dropna(inplace=True)
    
    # Select and scale features
    features_columns = [
        'price_change', 'price_momentum', 'price_volatility',
        'volume_change', 'volume_momentum',
        'consec_higher_highs', 'consec_lower_lows'
    ]
    
    scaler = StandardScaler()
    features = scaler.fit_transform(df[features_columns])
    
    return features, df

def label_market_regimes(ohlcv_data):
    """
    Label market regimes based on price and volume characteristics
    
    Returns:
    --------
    labels : numpy.ndarray
        Categorical labels for market regimes
    """
    df = ohlcv_data.copy()
    
    def classify_regime(row):
        # Trending Up: Consistent higher highs, higher volume
        if (row['High'] > row['High'].shift(1) and 
            row['Volume'] > row['Volume'].shift(1)):
            return 0  # Trending Up
        
        # Trending Down: Consistent lower lows, higher volume
        elif (row['Low'] < row['Low'].shift(1) and 
              row['Volume'] > row['Volume'].shift(1)):
            return 1  # Trending Down
        
        # Sideways/Consolidation: Narrow price range
        else:
            return 2  # Sideways
    
    labels = df.apply(classify_regime, axis=1)
    return labels.values

def train_market_regime_model(features, labels):
    """
    Train decision tree classifier with cross-validation and regularization
    
    Returns:
    --------
    model : DecisionTreeClassifier
        Trained decision tree model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Initialize model with regularization to prevent overfitting
    model = DecisionTreeClassifier(
        max_depth=5,           # Limit tree depth
        min_samples_split=20,  # Minimum samples to split
        min_samples_leaf=10,   # Minimum samples in leaf
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, features, labels, cv=5)
    
    # Print model performance
    y_pred = model.predict(X_test)
    print("Cross-validation scores:", cv_scores)
    print("Mean CV Score:", cv_scores.mean())
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return model

# Example Usage
# features, processed_df = engineer_market_regime_features(ohlcv_data)
# labels = label_market_regimes(ohlcv_data)
# model = train_market_regime_model(features, labels)
