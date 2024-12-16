# Market-Regime-Modeling-and-Rates-Prediction-Analysis

Here’s a detailed and professional **README** file for your GitHub repository:

---
  

This repository contains a comprehensive **Market Analysis Pipeline** that leverages machine learning techniques to analyze financial markets. The primary objectives include **market regime classification**, **clustering analysis**, **hidden market regime prediction**, and **rates prediction analysis**.  

## Features  

### **1. Feature Engineering**
- Extracts financial indicators such as:
  - Price Momentum
  - Moving Averages (50-day, 200-day)
  - Price Volatility
  - Volume Momentum and Trends
  - Consecutive Highs/Lows  
- Prepares the data for advanced analysis.  

### **2. Market Regime Classification**
- Classifies market behavior into:
  - **Trending Up**
  - **Trending Down**
  - **Sideways/Consolidation**  
- Uses a **Decision Tree Classifier** for predictions with metrics such as precision, recall, F1-score, and confusion matrix.  

### **3. Clustering Analysis**
- Groups trading days using **K-Means Clustering** based on:
  - Price Momentum
  - Volatility
  - Volume Trends  
- Automatically determines the optimal number of clusters using the **Elbow Method**.  

### **4. Hidden Markov Model (HMM)**
- Predicts hidden market regimes using **Gaussian HMM**:
  - Transition probabilities between states (e.g., Low Volatility, High Volatility).
  - State means and covariance matrices.  

### **5. Rates Prediction**
- Generates:
  - **Correlation Matrix:** Examines relationships between contracts.
  - **Ratio Matrix:** Computes price ratios between contracts using **Linear Regression**.  

### **6. Visualization**
- Produces insightful plots:
  - Clustering results.
  - HMM hidden state transitions.
  - Heatmap of the contract correlation matrix.  

---

## Installation  

### **1. Prerequisites**
Ensure the following libraries are installed:  
- `numpy`  
- `pandas`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  
- `hmmlearn`  
- `scipy`

Install the dependencies with:  
```bash
pip install -r requirements.txt
```

---

## Usage  

1. **Prepare Your Data**  
   - Provide two CSV files:  
     - `ohlcv_data.csv`: Contains OHLCV (Open, High, Low, Close, Volume) data.  
     - `time_sales_data.csv`: Includes detailed trade data with timestamps and volumes.  
   - An optional third CSV file `contracts_data.csv` is used for rates prediction.  

2. **Run the Script**  
   Execute the main script:  
   ```bash
   python market_analysis_pipeline.py
   ```

3. **Results**  
   - Outputs include:
     - Classification metrics.
     - Cluster summaries.
     - Hidden Markov Model parameters.
     - Correlation and ratio matrices.
   - Visualizations for clustering, HMM hidden states, and correlation heatmaps.  

---

## Example Outputs  

### **Market Regime Classification Results**
| Regime          | Precision | Recall | F1-Score |  
|------------------|-----------|--------|----------|  
| Trending Up      | 0.85      | 0.82   | 0.83     |  
| Trending Down    | 0.78      | 0.81   | 0.79     |  
| Sideways         | 0.80      | 0.77   | 0.78     |  

### **K-Means Clustering Analysis**
| Cluster        | Mean Return | Volatility | Observation Count |  
|-----------------|-------------|------------|-------------------|  
| Low Risk       | 0.02%       | 0.5%       | 85                |  
| Medium Risk    | 0.05%       | 1.2%       | 120               |  
| High Risk      | 0.10%       | 2.5%       | 95                |  

### **HMM Transition Matrix**
| From State \ To State | Low Volatility | High Volatility |  
|------------------------|----------------|------------------|  
| Low Volatility         | 0.85          | 0.15            |  
| High Volatility        | 0.25          | 0.75            |  

---

## Repository Structure  

```
├── market_analysis_pipeline.py   # Main Python script  
├── requirements.txt              # Dependencies  
├── data/                         # Sample data folder  
│   ├── ohlcv_data.csv  
│   ├── time_sales_data.csv  
│   ├── contracts_data.csv  
├── outputs/                      # Folder for results and visualizations  
├── README.md                     # Project documentation  
```

---

## Future Improvements  
- **Ensemble Models:** Combine multiple classifiers for improved regime classification.  
- **Deep Learning:** Explore RNNs or LSTMs for time-series forecasting.  
- **Interactive Visualizations:** Use libraries like Plotly for dynamic exploration of results.  

---

## License  
This project is licensed under the MIT License. See the `LICENSE` file for details.  

---

## Contact  
For questions or suggestions, please contact [Your Name] at [Your Email].  

---

Let me know if you'd like further refinements or additional sections!
