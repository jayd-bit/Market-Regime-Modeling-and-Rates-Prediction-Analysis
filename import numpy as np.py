import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, silhouette_score, mean_squared_error
from hmmlearn import hmm
from sklearn.linear_model import LinearRegression

class MarketAnalysisPipeline:
    def __init__(self, ohlcv_data, time_sales_data=None):
        """
        Initialize Market Analysis Pipeline
        
        Parameters:
        -----------
        ohlcv_data : pandas.DataFrame
            OHLCV (Open, High, Low, Close, Volume) data
        time_sales_data : pandas.DataFrame, optional
            Time and sales data for additional analysis
        """
        self.ohlcv_data = ohlcv_data
        self.time_sales_data = time_sales_data
        self.scaler = StandardScaler()

    def decision_tree_market_regime_classification(self):
        """
        Classify market regimes using Decision Tree
        
        Returns:
        --------
        model : DecisionTreeClassifier
            Trained decision tree model
        """
        # Feature engineering
        df = self.ohlcv_data.copy()
        df['price_change'] = df['Close'].pct_change()
        df['price_momentum'] = df['Close'].rolling(window=5).mean().pct_change()
        df['price_volatility'] = df['Close'].rolling(window=5).std()
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_momentum'] = df['Volume'].rolling(window=5).mean().pct_change()
        df['consec_higher_highs'] = (df['High'] > df['High'].shift(1)).rolling(window=3).sum()
        df['consec_lower_lows'] = (df['Low'] < df['Low'].shift(1)).rolling(window=3).sum()
        
        # Labeling market regimes
        def classify_regime(row):
            if (row['High'] > row['High'].shift(1) and row['Volume'] > row['Volume'].shift(1)):
                return 0  # Trending Up
            elif (row['Low'] < row['Low'].shift(1) and row['Volume'] > row['Volume'].shift(1)):
                return 1  # Trending Down
            else:
                return 2  # Sideways
        
        df['regime'] = df.apply(classify_regime, axis=1)
        
        # Prepare features
        features_columns = [
            'price_change', 'price_momentum', 'price_volatility',
            'volume_change', 'volume_momentum',
            'consec_higher_highs', 'consec_lower_lows'
        ]
        df_clean = df.dropna()
        X = self.scaler.fit_transform(df_clean[features_columns])
        y = df_clean['regime']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(max_depth=5, min_samples_split=20, min_samples_leaf=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        print("Decision Tree Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return model

    def market_clustering_analysis(self, max_clusters=10):
        """
        Perform market clustering analysis
        
        Parameters:
        -----------
        max_clusters : int, default=10
            Maximum number of clusters to consider
        
        Returns:
        --------
        optimal_clusters : int
            Optimal number of clusters
        cluster_labels : numpy.ndarray
            Cluster assignments
        """
        # Prepare clustering features
        daily_features = pd.DataFrame({
            'daily_return': self.ohlcv_data['Close'].pct_change(),
            'daily_volatility': self.ohlcv_data['Close'].rolling(window=5).std(),
            'volume_mean': self.ohlcv_data['Volume'].rolling(window=5).mean(),
            'volume_volatility': self.ohlcv_data['Volume'].rolling(window=5).std()
        }).dropna()
        
        features = self.scaler.fit_transform(daily_features)
        
        # Determine optimal clusters
        inertias = []
        silhouette_scores_list = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
            silhouette_scores_list.append(silhouette_score(features, kmeans.labels_))
        
        # Plot elbow and silhouette curves
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.plot(range(2, max_clusters + 1), inertias, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        
        plt.subplot(122)
        plt.plot(range(2, max_clusters + 1), silhouette_scores_list, marker='o')
        plt.title('Silhouette Score')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.tight_layout()
        plt.show()
        
        # Determine optimal clusters
        optimal_k = np.argmax(silhouette_scores_list) + 2
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Analyze clusters
        clustered_data = daily_features.copy()
        clustered_data['cluster'] = cluster_labels
        cluster_stats = clustered_data.groupby('cluster').agg({
            'daily_return': ['mean', 'std'],
            'daily_volatility': ['mean', 'std'],
            'volume_mean': ['mean', 'std']
        })
        
        print("\nCluster Statistics:")
        print(cluster_stats)
        
        return optimal_k, cluster_labels

    def hidden_markov_model_prediction(self, n_states=2):
        """
        Develop Hidden Markov Model for market regime prediction
        
        Parameters:
        -----------
        n_states : int, default=2
            Number of hidden states
        
        Returns:
        --------
        predicted_states : numpy.ndarray
            Predicted market regimes
        transition_matrix : numpy.ndarray
            Transition probabilities between states
        """
        # Prepare HMM features
        features_df = pd.DataFrame({
            'log_returns': np.log(self.ohlcv_data['Close'] / self.ohlcv_data['Close'].shift(1)),
            'volatility': self.ohlcv_data['Close'].rolling(window=5).std(),
            'volume_normalized': self.ohlcv_data['Volume'] / self.ohlcv_data['Volume'].rolling(window=5).mean()
        }).dropna()
        
        # Incorporate time and sales data if available
        if self.time_sales_data is not None:
            trade_intensity = self.time_sales_data.groupby(pd.Grouper(freq='5T')).size()
            trade_size_variance = self.time_sales_data.groupby(pd.Grouper(freq='5T'))['volume'].var()
            
            features_df['trade_intensity'] = trade_intensity
            features_df['trade_size_variance'] = trade_size_variance
        
        # Scale features
        features = self.scaler.fit_transform(features_df)
        
        # Train HMM
        hmm_model = hmm.GaussianHMM(
            n_components=n_states, 
            covariance_type='full',
            n_iter=100
        )
        hmm_model.fit(features)
        
        # Predict states
        predicted_states = hmm_model.predict(features)
        transition_matrix = hmm_model.transmat_
        
        print("\nHMM Transition Matrix:")
        print(transition_matrix)
        
        return predicted_states, transition_matrix

    def rates_prediction_analysis(self, window_days=15):
        """
        Perform rates prediction analysis
        
        Parameters:
        -----------
        window_days : int, default=15
            Number of days for correlation and ratio calculation
        
        Returns:
        --------
        correlation_matrix : pandas.DataFrame
            Correlation matrix of contract prices
        ratio_matrix : pandas.DataFrame
            Ratio matrix between contracts
        """
        # Correlation matrix
        correlation_matrix = self.ohlcv_data.rolling(window=window_days).corr()
        correlation_matrix = correlation_matrix.iloc[-len(self.ohlcv_data.columns):]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title(f'Correlation Matrix ({window_days} Days Window)')
        plt.show()
        
        # Ratio matrix using linear regression
        contracts = self.ohlcv_data.columns
        ratio_matrix = pd.DataFrame(
            np.zeros((len(contracts), len(contracts))),
            index=contracts,
            columns=contracts
        )
        
        for i in range(len(contracts)):
            for j in range(len(contracts)):
                if i != j:
                    X = self.ohlcv_data[contracts[i]].values.reshape(-1, 1)
                    y = self.ohlcv_data[contracts[j]].values
                    
                    reg = LinearRegression().fit(X, y)
                    ratio_matrix.iloc[i, j] = reg.coef_[0]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(ratio_matrix, annot=True, cmap='coolwarm')
        plt.title(f'Ratio Matrix ({window_days} Days Window)')
        plt.show()
        
        return correlation_matrix, ratio_matrix

def main():
    # Load your OHLCV data (replace with actual data loading)
    ohlcv_data = pd.read_csv('your_ohlcv_data.csv')
    time_sales_data = pd.read_csv('your_time_sales_data.csv')  # Optional
    
    # Initialize pipeline
    pipeline = MarketAnalysisPipeline(ohlcv_data, time_sales_data)
    
    # Run analyses
    pipeline.decision_tree_market_regime_classification()
    pipeline.market_clustering_analysis()
    pipeline.hidden_markov_model_prediction()
    pipeline.rates_prediction_analysis()

if __name__ == "__main__":
    main()