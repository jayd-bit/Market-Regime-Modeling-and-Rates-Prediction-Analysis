import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

class MarketRegimeHMM:
    def __init__(self, n_states=2):
        """
        Initialize Hidden Markov Model for market regimes
        
        Parameters:
        -----------
        n_states : int, default=2
            Number of hidden states (risk-on/risk-off)
        """
        self.n_states = n_states
        self.model = None
        self.scaler = StandardScaler()
    
    def _prepare_features(self, ohlcv_data, time_sales_data=None):
        """
        Prepare features for HMM training
        
        Parameters:
        -----------
        ohlcv_data : pandas.DataFrame
            OHLCV data
        time_sales_data : pandas.DataFrame, optional
            Time and sales data for enhanced feature engineering
        
        Returns:
        --------
        features : numpy.ndarray
            Scaled features for HMM
        """
        # Basic OHLCV features
        features_df = pd.DataFrame({
            'log_returns': np.log(ohlcv_data['Close'] / ohlcv_data['Close'].shift(1)),
            'volatility': ohlcv_data['Close'].rolling(window=5).std(),
            'volume_normalized': ohlcv_data['Volume'] / ohlcv_data['Volume'].rolling(window=5).mean()
        }).dropna()
        
        # Incorporate Time and Sales data if available
        if time_sales_data is not None:
            # Example: Add trade intensity and trade size variance
            trade_intensity = time_sales_data.groupby(pd.Grouper(freq='5T')).size()
            trade_size_variance = time_sales_data.groupby(pd.Grouper(freq='5T'))['volume'].var()
            
            features_df['trade_intensity'] = trade_intensity
            features_df['trade_size_variance'] = trade_size_variance
        
        # Scale features
        features = self.scaler.fit_transform(features_df)
        return features
    
    def train(self, ohlcv_data, time_sales_data=None):
        """
        Train Hidden Markov Model
        
        Parameters:
        -----------
        ohlcv_data : pandas.DataFrame
            OHLCV data
        time_sales_data : pandas.DataFrame, optional
            Time and sales data
        """
        features = self._prepare_features(ohlcv_data, time_sales_data)
        
        # Initialize and train HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states, 
            covariance_type='full',
            n_iter=100
        )
        
        self.model.fit(features)
    
    def predict_regime(self, new_data):
        """
        Predict most likely market regime
        
        Returns:
        --------
        predicted_states : numpy.ndarray
            Predicted market regime for each time point
        """
        features = self._prepare_features(new_data)
        predicted_states = self.model.predict(features)
        return predicted_states
    
    def analyze_regime_transitions(self):
        """
        Analyze transition probabilities between regimes
        
        Returns:
        --------
        transition_matrix : numpy.ndarray
            Transition probabilities between hidden states
        """
        return self.model.transmat_

# Example Usage
# hmm_model = MarketRegimeHMM(n_states=2)
# hmm_model.train(ohlcv_data, time_sales_data)
# predicted_states = hmm_model.predict_regime(new_data)
# transition_matrix = hmm_model.analyze_regime_transitions()
