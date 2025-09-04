# Outlier Detection Algorithm Decision Tree

### 1. **Data Characteristics Assessment**
   - **Sample Size (N)**
   - **Dimensionality (D)**
   - **Outlier Type**: Extreme vs Near outliers
   - **Time Constraints**: Real-time vs Batch processing
   - **Data Distribution**: Normal, Skewed, Uniform, Mixed
   - **Multimodal/Non-linearity**: Simple vs Complex patterns
   - **Label Availability**: Supervised, Semi-supervised, Unsupervised

### 2. **Enhanced Decision Tree Structure**

```
Start : Outlier Detection Algorithm Selection
│
├─── Do you have labeled outlier data or history?
│    │
│    ├─── YES (Supervised Learning Available)
│    │    │
│    │    ├─── Large dataset (N > 1000)?
│    │    │    ├─── YES -> **Supervised ML** (Random Forest, SVM, Neural Networks)
│    │    │    │         Use labeled data for training classification model
│    │    │    └─── NO -> **Semi-supervised approach**
│    │    │              Primary: **One-Class SVM **
│    │    │              Alternative: **Feature Bagging** with partial labels
│    │    │
│    │    └─── NO -> Proceed to unsupervised decision tree below
│    │
│    └─── NO (Unsupervised Learning Required)
│         │
│         ├─── What is your data distribution?
│         │    │
│         │    ├─── **Normal/Gaussian Distribution**
│         │    │    │
│         │    │    ├─── Single mode, linear relationships?
│         │    │    │    ├─── YES (Simple Gaussian)
│         │    │    │    │    │
│         │    │    │    │    ├─── N < 500 & D ≤ 8?
│         │    │    │    │    │    ├─── Extreme outliers -> **Z-Score**, **IQR**
│         │    │    │    │    │    │                      **Mahalanobis** for multivariate
│         │    │    │    │    │    └─── Near outliers -> **LOF**
│         │    │    │    │    │
│         │    │    │    │    └─── Larger datasets -> **Elliptic Envelope**
│         │    │    │    │                           Assumes Gaussian, handles correlations
│         │    │    │    │
│         │    │    │    └─── NO (Complex Gaussian patterns)
│         │    │    │         -> **Mixture models approach**
│         │    │    │           Primary: **HBOS** 
│         │    │    │           Alternative: **PCA** for dimensionality reduction
│         │    │    │
│         │    │    └─── **Multimodal Gaussian**
│         │    │         │
│         │    │         ├─── Known cluster structure?
│         │    │         │    ├─── YES -> **CBLOF**
│         │    │         │    │         Cluster-based detection for multimodal
│         │    │         │    └─── NO -> **LOF**
│         │    │         │              Density-based, handles multiple modes
│         │    │         │
│         │    │         └─── High dimension (D > 25)?
│         │    │              -> **Feature Bagging** 
│         │    │                Subspace methods for multimodal high-D data
│         │    │
│         │    ├─── **Skewed/Heavy-tailed Distribution**
│         │    │    │
│         │    │    ├─── Moderate skew?
│         │    │    │    ├─── YES -> **IQR** 
│         │    │    │    │         Alternative: **Cook's Distance**
│         │    │    │    └─── NO (Extreme skew) -> **HBOS **
│         │    │    │                            Histogram-based, handles any distribution
│         │    │    │
│         │    │    └─── Non-linear relationships?
│         │    │         ├─── YES -> **Isolation Forest**
│         │    │         │         Tree-based, captures non-linear patterns
│         │    │         └─── NO -> **Modified Z-Score** or **IQR**
│         │    │
│         │    ├─── **Uniform/Unknown Distribution**
│         │    │    │
│         │    │    ├─── Non-parametric approach required
│         │    │    │    │
│         │    │    │    ├─── Simple patterns -> **HBOS**
│         │    │    │    │                     
│         │    │    │    │
│         │    │    │    ├─── Complex patterns -> **KNN**
│         │    │    │    │                      Distance-based, no distribution assumptions
│         │    │    │    │
│         │    │    │    └─── Local density matters -> **LOF**
│         │    │    │                                Local density estimation
│         │    │    │
│         │    │    └─── High-dimensional uniform -> **PCA**, **Isolation Forest**
│         │    │                                   Subspace projection approach
│         │    │
│         │    └─── **Mixed/Complex Distributions**
│         │         │
│         │         ├─── Multiple distribution types?
│         │         │    -> **Ensemble approach**:
│         │         │      1. **HBOS** as baseline
│         │         │      2. **Isolation Forest** for non-linearity
│         │         │      3. **LOF** for local patterns
│         │         │      Combine results using voting or stacking
│         │         │
│         │         └─── Unknown complexity -> **HBOS**
│         │
│         ├─── **Multimodal/Non-linearity Assessment**
│         │    │
│         │    ├─── **Simple Linear Relationships**
│         │    │    │
│         │    │    ├─── Single cluster/mode -> Statistical methods preferred
│         │    │    │    │
│         │    │    │    ├─── Normal data -> **Z-Score**, **Mahalanobis**
│         │    │    │    ├─── Skewed data -> **IQR**, **Cook's Distance**
│         │    │    │    └─── High-D -> **PCA**, **Robust Covariance (MCD)**
│         │    │    │
│         │    │    └─── Multiple clusters -> **CBLOF**
│         │    │                            **Robust Covariance** for Gaussian mixtures
│         │    │
│         │    ├─── **Moderate Non-linearity**
│         │    │    │
│         │    │    ├─── Density-based patterns -> **LOF**
│         │    │    │
│         │    │    ├─── Distance-based patterns -> **KNN**
│         │    │    │                             **Angle-based Outlier**
│         │    │    │
│         │    │    └─── Tree-like separability -> **Isolation Forest**
│         │    │                                  Handles moderate non-linearity well
│         │    │
│         │    └─── **High Non-linearity/Complex Patterns**
│         │         │
│         │         ├─── Time constraint critical?
│         │         │    ├─── YES -> **HBOS**
│         │         │    │         Works regardless of complexity
│         │         │    └─── NO -> **Ensemble of multiple methods**:
│         │         │              - **Isolation Forest** (tree-based non-linearity)
│         │         │              - **LOF** (local density patterns)  
│         │         │              - **KNN ** (distance-based patterns)
│         │         │
│         │         ├─── Subspace patterns -> **Feature Bagging**
│         │         │                       **COPOD** for copula-based dependencies
│         │         │
│         │         └─── Unknown complexity -> **HBOS** (Universal robust choice)
│         │
│         └─── **Final Decision by Scale and Performance**
│              │
│              ├─── N < 500 (Small datasets)
│              │    ├─── D ≤ 8 -> Choose based on distribution analysis above but mostly **IQR**, **Cook's D** for extreme outliers and **LOF**, **Isolation Forest** for near outliers
│              │    └─── D > 8
|              |         ├─── Time critical?
│              │         |    ├─── Yes -> Z-score
|              |         |    └─── No -> Robust Covariance
|              |         |   
|              |         └─── Near outliers : One-Class SVM  
|              |     
│              ├─── 500 ≤ N ≤ 5000 (Medium datasets)  
│              │    ├─── Time critical -> **IQR**, **Z-score**, **HBOS**, **PCA**
│              │    └─── Accuracy priority -> **KNN**, **LOF**
│              │
│              └─── N > 5000 (Large datasets)
│                   ├─── D ≤ 30 -> **KNN**, **LOF**, **Isolation Forest**
│                   └─── D > 30 -> **HBOS**, **PCA**, **Feature Bagging**
```

## Algorithm Selection Matrix by Data Characteristics

| Data Type                | Distribution | Multimodal | Non-linear | Primary Choice          | Alternative       | Avoid               |
| ------------------------ | ------------ | ---------- | ---------- | ----------------------- | ----------------- | ------------------- |
| **Labeled Historical**   | Any          | Any        | Any        | Supervised ML           | One-Class SVM     | Pure unsupervised   |
| **Gaussian Single Mode** | Normal       | No         | No         | IQR/Z-Score/Mahalanobis | Robust Covariance | HBOS                |
| **Gaussian Multi-Mode**  | Normal       | Yes        | No         | CBLOF                   | LOF               | Statistical methods |
| **Skewed Light**         | Skewed       | No         | No         | IQR                     | Cook's Distance   | Z-Score             |
| **Skewed Heavy**         | Heavy-tail   | Variable   | Variable   | HBOS                    | Isolation Forest  | Statistical methods |
| **Uniform/Unknown**      | Uniform      | Unknown    | Unknown    | HBOS                    | KNN               | Parametric methods  |
| **Complex Mixed**        | Mixed        | Yes        | Yes        | Ensemble                | HBOS              | Single method       |
| **High-D Simple**        | Any          | No         | No         | PCA                     | Robust Covariance | LOF, KNN            |
| **High-D Complex**       | Any          | Yes        | Yes        | Feature Bagging         | HBOS              | Statistical methods |