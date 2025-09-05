from time import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import qmc

import statsmodels.api as sm

from pymoo.problems import get_problem

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDOneClassSVM, LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split

from pyod.models.iforest import IForest
from pyod.models.copod import COPOD
from pyod.models.abod import ABOD
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from pyod.models.loci import LOCI

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class Benchmark:
    def __init__(self, name: str, dim: int = None):
        self.name = name.lower()
        self.dim = dim
        
        if self.name == "rosenbrock":
            if self.dim is None:
                raise ValueError("Set dim for Rosenbrock")
            self.problem = get_problem("rosenbrock", n_var=self.dim)
        elif self.name == "kursawe":
            self.problem = get_problem("kursawe")
            self.dim = self.problem.n_var
        elif self.name == "sphere":
            if self.dim is None:
                self.dim = 2
            self.problem = get_problem("sphere", n_var=self.dim)
        else:
            self.problem = get_problem(self.name, n_var=self.dim)
            if self.dim is None:
                self.dim = self.problem.n_var
        
        self.bounds = np.vstack([self.problem.xl, self.problem.xu]).T

    # DoE generators
    def sample_sobol(self, n_points: int, scramble: bool = True):
        sampler = qmc.Sobol(d=self.dim, scramble=scramble)
        sample = sampler.random(n_points)
        return qmc.scale(sample, self.bounds[:,0], self.bounds[:,1])
    
    def sample_halton(self, n_points: int, scramble: bool = True):
        sampler = qmc.Halton(d=self.dim, scramble=scramble)
        sample = sampler.random(n_points)
        return qmc.scale(sample, self.bounds[:,0], self.bounds[:,1])
    
    def sample_lhs(self, n_points: int, criterion: str = "random-cd", seed: int = 42): # values of criterion : 'random-cd', 'lloyd'
        sampler = qmc.LatinHypercube(d=self.dim, optimization=criterion, seed=seed)
        sample = sampler.random(n_points)
        print("LHS discrepancy:", qmc.discrepancy(sample))
        return qmc.scale(sample, self.bounds[:,0], self.bounds[:,1])

    def evaluate(self, X: np.ndarray):
        return self.problem.evaluate(X)
    
    def inject_outliers(self, X, Y, frac: float = 0.05, scale_factor: float = 0.2):
        """Inject near-outliers into Y (output). Returns noisy Y and indices of injected outliers."""
        n_outliers = max(1, int(frac * X.shape[0]))
        idx = np.random.choice(X.shape[0], n_outliers, replace=False)
        Y_out = Y.copy()
        scale = np.std(Y, axis=0) * scale_factor
        if Y_out.ndim == 1:
            Y_out[idx] = Y_out[idx] + np.random.normal(0, scale, size=n_outliers)
        else:
            Y_out[idx, 0] = Y_out[idx, 0] + np.random.normal(0, scale, size=n_outliers)
        return Y_out, idx
    


class StatisticalOutlierDetection:
    def z_score(self, data, threshold: float = 3.0):
        z_scores = np.abs(stats.zscore(data, axis=0))
        scores = np.max(z_scores, axis=1)
        return scores > threshold, scores

    def iqr(self, data, factor: float = 1.5):
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        mask = np.any((data < lower) | (data > upper), axis=1)
        dist = np.max(np.maximum(0, data - upper) + np.maximum(0, lower - data), axis=1)
        return mask, dist

    def leverage(self, X, threshold: float = None):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_design = np.column_stack([np.ones(X.shape[0]), X])
        H = X_design @ np.linalg.pinv(X_design.T @ X_design) @ X_design.T
        leverages = np.diag(H)
        if threshold is None:
            threshold = 2 * X_design.shape[1] / X_design.shape[0]
        return leverages > threshold, leverages

    def cooks_distance(self, X, y, threshold: float = None):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = y.ravel()
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        X_design = np.column_stack([np.ones(X.shape[0]), X])
        H = X_design @ np.linalg.pinv(X_design.T @ X_design) @ X_design.T
        leverages = np.diag(H)
        p = X_design.shape[1]
        n = X_design.shape[0]
        mse = np.sum(residuals**2) / max(1, (n - p))
        cooks_d = (residuals**2 / (p * mse)) * (leverages / (1 - leverages)**2)
        if threshold is None:
            threshold = 4 / n
        return cooks_d > threshold, cooks_d

    def mahalanobis(self, X, threshold: float = None):
        cov = EmpiricalCovariance().fit(X)
        m_dist = cov.mahalanobis(X)
        if threshold is None:
            # Use a high percentile as a generic cutoff; ROC will use the full score anyway
            threshold = np.percentile(m_dist, 97.5)
        return m_dist > threshold, m_dist
    

class MLOutlierDetection:

    def __init__(self):
        self.scaler = StandardScaler()

    def elliptic_envelope(self, X, contamination: float = 0.1):
        model = EllipticEnvelope(contamination=contamination, random_state=42)
        pred = model.fit_predict(X)
        return pred == -1, -model.decision_function(X)

    def isolation_forest(self, X, contamination: float = 0.1):
        model = IsolationForest(contamination=contamination, random_state=42)
        pred = model.fit_predict(X)
        return pred == -1, -model.decision_function(X)

    def lof(self, X, contamination: float = 0.1, n_neighbors: int = 20):
        model = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors)
        pred = model.fit_predict(X)
        return pred == -1, -model.negative_outlier_factor_

    def one_class_svm(self, X, nu: float = 0.1):
        model = OneClassSVM(gamma="scale", nu=nu)
        pred = model.fit_predict(X)
        return pred == -1, -model.decision_function(X)

    def sgd_one_class_svm(self, X, nu: float = 0.1):
        try:
            model = SGDOneClassSVM(nu=nu, random_state=42)
            pred = model.fit_predict(X)
            return pred == -1, -model.decision_function(X)
        except Exception:
            # If not available in the local sklearn version, return blanks
            return np.zeros(X.shape[0], dtype=bool), np.zeros(X.shape[0])
        
    def dbscan(self, X, eps: float = 0.5, min_samples: int = 5):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        outliers = labels == -1
        # Since DBSCAN doesn't provide anomaly scores, I set score=1 for outliers, 0 otherwise
        scores = outliers.astype(int)
        return outliers, scores
    

scale_factor_extreme = 10.0
scale_factor_near = 0.5

y_true = {}
y_score = {"Z-Score (Extreme)": {}, "IQR (Extreme)": {}, "Leverage (Extreme)": {}, "Cook's Distance (Extreme)": {}, "Mahalanobis (Extreme)": {},
           "Elliptic Envelope (Extreme)": {}, "Isolation Forest (Extreme)": {}, "LOF (Extreme)": {}, "One-Class SVM (Extreme)": {}, "SGD One-Class SVM (Extreme)": {}, "DBSCAN (Extreme)": {},
           "Z-Score (Near)": {}, "IQR (Near)": {}, "Leverage (Near)": {}, "Cook's Distance (Near)": {}, "Mahalanobis (Near)": {},
           "Elliptic Envelope (Near)": {}, "Isolation Forest (Near)": {}, "LOF (Near)": {}, "One-Class SVM (Near)": {}, "SGD One-Class SVM (Near)": {}, "DBSCAN (Near)": {}}
model_names = ["Z-Score (Extreme)", "IQR (Extreme)", "Leverage (Extreme)", "Cook's Distance (Extreme)", "Mahalanobis (Extreme)",
               "Elliptic Envelope (Extreme)", "Isolation Forest (Extreme)", "LOF (Extreme)", "One-Class SVM (Extreme)", "SGD One-Class SVM (Extreme)", "DBSCAN (Extreme)",
               "Z-Score (Near)", "IQR (Near)", "Leverage (Near)", "Cook's Distance (Near)", "Mahalanobis (Near)",
               "Elliptic Envelope (Near)", "Isolation Forest (Near)", "LOF (Near)", "One-Class SVM (Near)", "SGD One-Class SVM (Near)", "DBSCAN (Near)"]

def evaluate_detection(true_idx, pred_mask, scores, method_name, duration, bench_name, dim):
    y_true = np.zeros(len(pred_mask), dtype=int)
    y_true[true_idx] = 1
    y_pred = pred_mask.astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Normalisation des scores pour AUC
    scores = np.asarray(scores)
    if np.max(scores) > np.min(scores):
        scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    else:
        scores_norm = scores
    try:
        auc = roc_auc_score(y_true, scores_norm)
    except:
        auc = np.nan

    return {
        "benchmark": bench_name,
        "dim": dim,
        "method": method_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "time": duration
    }

def run_detection(detector, X, Y, idx_outliers, bench_name, dim, label):
    t0 = time.time()
    mask, scores = detector(X, Y)
    duration = time.time() - t0
    return evaluate_detection(idx_outliers, mask, scores, label, duration, bench_name, dim)

def run_detection_pyod(model, X, Y, idx_outliers, bench_name, dim, label):
    from pyod.utils.utility import standardizer

    X_aug = np.hstack([X, Y.reshape(-1, 1)])
    X_norm, _ = standardizer(X_aug, X_aug)

    t0 = time.time()
    model.fit(X_norm)
    scores = model.decision_function(X_norm)
    preds = model.predict(X_norm)
    duration = time.time() - t0

    return evaluate_detection(idx_outliers, preds == 1, scores, label, duration, bench_name, dim)

def run_benchmarks(benchmarks, dims, detectors_stat, detectors_ml, detectors_pyod, N=500):
    global results_list

    for bench_name in benchmarks:
        for dim in dims:
            # Instancier le benchmark
            bench = Benchmark(bench_name, dim=dim)
            X = bench.sample_lhs(N)
            Y = bench.evaluate(X)
            Y = np.asarray(Y).reshape(-1)

            # Injection artificielle d’outliers
            Y_noisy, idx = bench.inject_outliers(X, Y, n_outliers=int(0.05 * N), mode="Y")

            # Tester statistical detectors
            for name, det in detectors_stat.items():
                res = run_detection(det, X, Y_noisy, idx, bench_name, dim, name)
                results_list.append(res)

            # Tester ML detectors
            for name, det in detectors_ml.items():
                res = run_detection(det, X, Y_noisy, idx, bench_name, dim, name)
                results_list.append(res)

            # Tester PyOD detectors
            for name, model in detectors_pyod.items():
                res = run_detection_pyod(model, X, Y_noisy, idx, bench_name, dim, name)
                results_list.append(res)

    return pd.DataFrame(results_list)

# Définir tes détecteurs
detectors_stat = {
    "Z-Score": lambda X, Y: StatisticalOutlierDetection().z_score(Y),
    "IQR": lambda X, Y: StatisticalOutlierDetection().iqr(Y),
}

detectors_ml = {
    "Isolation Forest": lambda X, Y: MLOutlierDetection().isolation_forest(np.hstack([X, Y.reshape(-1,1)])),
}

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
detectors_pyod = {
    "PyOD IForest": IForest(contamination=0.05),
    "PyOD LOF": LOF(contamination=0.05)
}

# Lancer plusieurs benchmarks et dimensions
df_results = run_benchmarks(
    benchmarks=["rosenbrock", "sphere"],
    dims=[5, 20],
    detectors_stat=detectors_stat,
    detectors_ml=detectors_ml,
    detectors_pyod=detectors_pyod,
    N=500
)

print(df_results.head())

# Exemple: matrice de corrélation des scores AUC
df_pivot = df_results.pivot_table(index=["benchmark","dim"], columns="method", values="roc_auc")
print(df_pivot.corr())


def run_detection(X, y_noisy, idx, bench_name, contamination_rate, stat, ml, y_score, y_true, label_suffix="", eps: float = 0.5, min_samples: int = 5):
    """Apply statistical + ML outlier detection methods and store results."""
 
    # Statistical on y only
    t0 = time()
    mask, scores = stat.z_score(y_noisy.reshape(-1,1))
    t1 = time()
    y_true_pred, y_score_norm = evaluate_detection(idx, mask, scores, "Z-Score" + label_suffix, duration=t1 - t0)
    y_score["Z-Score" + label_suffix][bench_name] = y_score_norm
    y_true[bench_name] = y_true_pred
 
    t0 = time()
    mask, scores = stat.iqr(y_noisy.reshape(-1,1), factor=1.5)
    t1 = time()
    y_true_pred, y_score_norm = evaluate_detection(idx, mask, scores, "IQR" + label_suffix, duration=t1 - t0)
    y_score["IQR" + label_suffix][bench_name] = y_score_norm
    y_true[bench_name] = y_true_pred
 
    # Statistical on joint [X, y]
    X_aug = np.hstack([X, y_noisy.reshape(-1,1)])
    t0 = time()
    mask, scores = stat.mahalanobis(X_aug)
    t1 = time()
    y_true_pred, y_score_norm = evaluate_detection(idx, mask, scores, "Mahalanobis" + label_suffix, duration=t1 - t0)
    y_score["Mahalanobis" + label_suffix][bench_name] = y_score_norm
    y_true[bench_name] = y_true_pred
 
    t0 = time()
    mask, scores = stat.leverage(X)
    t1 = time()
    y_true_pred, y_score_norm = evaluate_detection(idx, mask, scores, "Leverage" + label_suffix, duration=t1 - t0)
    y_score["Leverage" + label_suffix][bench_name] = y_score_norm
    y_true[bench_name] = y_true_pred
 
    t0 = time()
    mask, scores = stat.cooks_distance(X, y_noisy)
    t1 = time()
    y_true_pred, y_score_norm = evaluate_detection(idx, mask, scores, "Cook's Distance" + label_suffix, duration=t1 - t0)
    y_score["Cook's Distance" + label_suffix][bench_name] = y_score_norm
    y_true[bench_name] = y_true_pred
 
    # ML methods on joint [X, y]
    X_aug = ml.scaler.fit_transform(X_aug)
 
    t0 = time()
    mask, scores = ml.elliptic_envelope(X_aug, contamination=contamination_rate)
    t1 = time()
    y_true_pred, y_score_norm = evaluate_detection(idx, mask, scores, "Elliptic Envelope" + label_suffix, duration=t1 - t0)
    y_score["Elliptic Envelope" + label_suffix][bench_name] = y_score_norm
    y_true[bench_name] = y_true_pred
 
    t0 = time() 
    mask, scores = ml.isolation_forest(X_aug, contamination=contamination_rate)
    t1 = time()
    y_true_pred, y_score_norm = evaluate_detection(idx, mask, scores, "Isolation Forest" + label_suffix, duration=t1 - t0)
    y_score["Isolation Forest" + label_suffix][bench_name] = y_score_norm
    y_true[bench_name] = y_true_pred
 
    t0 = time()
    mask, scores = ml.lof(X_aug, contamination=contamination_rate, n_neighbors=max(20, int(contamination_rate * len(X))))
    t1 = time()
    y_true_pred, y_score_norm = evaluate_detection(idx, mask, scores, "LOF" + label_suffix, duration=t1 - t0)
    y_score["LOF" + label_suffix][bench_name] = y_score_norm
    y_true[bench_name] = y_true_pred
 
    t0 = time()
    mask, scores = ml.one_class_svm(X_aug, nu=contamination_rate)
    t1 = time()
    y_true_pred, y_score_norm = evaluate_detection(idx, mask, scores, "One-Class SVM" + label_suffix, duration=t1 - t0)
    y_score["One-Class SVM" + label_suffix][bench_name] = y_score_norm
    y_true[bench_name] = y_true_pred
 
    t0 = time()
    mask, scores = ml.sgd_one_class_svm(X_aug, nu=contamination_rate)
    t1 = time()
    y_true_pred, y_score_norm = evaluate_detection(idx, mask, scores, "SGD One-Class SVM" + label_suffix, duration=t1 - t0)
    y_score["SGD One-Class SVM" + label_suffix][bench_name] = y_score_norm
    y_true[bench_name] = y_true_pred

    t0 = time()
    mask, scores = ml.dbscan(X_aug, eps=eps, min_samples=min_samples)
    t1 = time()
    y_true_pred, y_score_norm = evaluate_detection(idx, mask, scores, "DBSCAN" + label_suffix, duration=t1 - t0)
    y_score["DBSCAN" + label_suffix][bench_name] = y_score_norm
    y_true[bench_name] = y_true_pred


def run_detection_pyod(X, y_noisy, idx, n_clusters: int = 8, contamination_rate: float=0.5):
    
    # ground truth: 1 = outlier, 0 = normal
    y_true = np.zeros(X.shape[0], dtype=int)
    y_true[idx] = 1

    random_state = np.random.RandomState(42)

    classifiers = {
        "Angle-based Outlier Detector": ABOD(contamination=contamination_rate),
        'Feature Bagging': FeatureBagging(contamination=contamination_rate, random_state=random_state),
        'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=contamination_rate),
        "Isolation Forest": IForest(contamination=contamination_rate, random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=contamination_rate),
        "LOF": LOF(contamination=contamination_rate),
        #"LOCI": LOCI(contamination=contamination_rate),
        'Minimum Covariance Determinant (MCD)': MCD(contamination=contamination_rate, random_state=random_state),
        'One-class SVM (OCSVM)': OCSVM(contamination=contamination_rate),
        'Principal Component Analysis (PCA)': PCA(contamination=contamination_rate, random_state=random_state),
        "COPOD": COPOD(),
        'Cluster-based Local Outlier Factor': CBLOF(n_clusters= n_clusters, contamination=contamination_rate, check_estimator=False, random_state=random_state),
        #"DBSCAN" : DBSCAN()
    }

    # 60% data for training and 40% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.4, random_state=random_state)

    X_aug_train = np.hstack([X_train, y_train.reshape(-1,1)])
    X_aug_test = np.hstack([X_test, y_test.reshape(-1,1)])

    # X_aug_train = np.hstack([X_train, y_noisy[:len(X_train)].reshape(-1,1)])
    # X_aug_test = np.hstack([X_test, y_noisy[len(X_train):].reshape(-1,1)])

    # X_aug_train = X_train
    # X_aug_test = X_test

    # Standardize data
    X_train_norm, X_test_norm = standardizer(X_aug_train, X_aug_test)

    for clf_name, clf in classifiers.items():
        try:
            t0 = time()
            clf.fit(X_train_norm)
            test_scores = clf.decision_function(X_test_norm)
            t1 = time()
            duration = round(t1 - t0, 4)

            roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
            prn = round(precision_n_scores(y_test, test_scores), ndigits=4)
            
            #y_pred = clf.predict(X_test_norm)
            #mask = y_pred == 1
            
            print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, '
                    'execution time: {duration}s'.format(
                    clf_name=clf_name, roc=roc, prn=prn, duration=duration))
        except Exception as e:
            print(f"Erreur avec {clf_name}: {e}")
            continue

stat = StatisticalOutlierDetection()
ml = MLOutlierDetection()


def run_detection_pipeline(X, y, bench: Benchmark, stat: StatisticalOutlierDetection, ml: MLOutlierDetection, N: int = 500, contamination_rate: float = 0.05, n_clusters: int = 8, eps: float = 0.5):
    y_noisy_extreme, idx_extreme = bench.inject_outliers(X, y, frac=contamination_rate, scale_factor=scale_factor_extreme) # Extreme outliers injection
    y_noisy_near, idx_near = bench.inject_outliers(X, y, frac=contamination_rate, scale_factor=scale_factor_near) # Near outliers injection
    print(f"Injected outliers: {len(idx_extreme)} / {N}")

    print(f"\n=== Benchmark: {bench.name.title()} ({bench.dim}D) with Extreme Outliers ===")
    run_detection(X, y_noisy_extreme, idx_extreme, bench.name, contamination_rate,
                                    stat, ml, y_score, y_true, label_suffix=" (Extreme)", eps=eps, min_samples=(bench.dim**2))
    print("\n--- Now running PyOD methods ---")
    run_detection_pyod(X, y_noisy_extreme, idx_extreme, contamination_rate=contamination_rate)
    
    print(f"\n\n=== Benchmark: {bench.name.title()} ({bench.dim}D) with Near Outliers ===")
    run_detection(X, y_noisy_near, idx_near, bench.name, contamination_rate,
                                    stat, ml, y_score, y_true, label_suffix=" (Near)", eps=eps, min_samples=(bench.dim**2))
    print("\n--- Now running PyOD methods ---")
    run_detection_pyod(X, y_noisy_near, idx_near, n_clusters = n_clusters, contamination_rate=contamination_rate)


bench = Benchmark("rosenbrock", dim=8)
N = 300
X = bench.sample_lhs(N)
y = bench.evaluate(X)
y = np.asarray(y).reshape(-1)
 
contamination_rate = 0.05

# plot_k_distance_graph(X, k=5)

run_detection_pipeline(X, y, bench, stat, ml, N=N, contamination_rate=contamination_rate, eps=0.5)