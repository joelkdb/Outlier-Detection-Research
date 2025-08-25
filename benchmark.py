import numpy as np

from scipy.stats import qmc

from pymoo.problems import get_problem
 
 
class Benchmark:

    def __init__(self, name: str, dim: int = None):

        self.name = name.lower()
        self.dim = dim

        # Instanciation of benchmark via pymoo
        if self.name == "rosenbrock":
            if self.dim is None:
                raise ValueError("Set the dimension for Rosenbrock")
            self.problem = get_problem("rosenbrock", n_var=self.dim)
        elif self.name == "branin":
            self.problem = get_problem("branin")
            self.dim = self.problem.n_var
        elif self.name == "kursawe":
            self.problem = get_problem("kursawe")
            self.dim = self.problem.n_var
        elif self.name == "hartmann":
            if self.dim not in (3, 6):
                raise ValueError("Hartmann available in 3D or 6D in pymoo")
            self.problem = get_problem(f"hartmann{self.dim}")
        else:
            raise ValueError(f"Unknown Benchmark : {self.name}.")

        # bounds
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

    def sample_lhs(self, n_points: int, criterion: str = "maximin"):
        sampler = qmc.LatinHypercube(d=self.dim, optimization=criterion)
        sample = sampler.random(n_points)
        return qmc.scale(sample, self.bounds[:,0], self.bounds[:,1])


    # Evaluation via pymoo
    def evaluate(self, X: np.ndarray):
        return self.problem.evaluate(X)

    # Outlier injection
    def inject_outliers(self, X, Y, n_outliers=10, mode="X"):

        """
        Inject outliers in X, Y or both.

        Parameters
        ----------
        X : np.ndarray
            Design parameters (n,d)
        Y : np.ndarray
            Output parameters (n,m)
        n_outliers : int
            Number of outliers to inject

        mode : str
            "X" -> outliers in X
            "Y" -> outliers in Y
            "both" -> outliers in X et Y

        Returns
        -------
        X_out, Y_out : data with outliers injected

        """

        X_out, Y_out = X.copy(), Y.copy()
        n = X.shape[0]

        idx = np.random.choice(n, n_outliers, replace=False)
        if mode in ["X", "both"]:
            # Génère des valeurs très éloignées (ex: ±10x bornes)

            span = self.bounds[:,1] - self.bounds[:,0]
            noise = np.random.uniform(-5, 5, size=(n_outliers, self.dim)) * span
            X_out[idx] = X_out[idx] + noise

        if mode in ["Y", "both"]:
            # Ajoute du bruit gaussien fort sur Y
            scale = np.std(Y, axis=0) * 10.0
            noise = np.random.normal(0, scale, size=Y_out[idx].shape)
            Y_out[idx] = Y_out[idx] + noise

        return X_out, Y_out
 
 


# Test code for the Benchmark class
if __name__ == "__main__":
    # Rosenbrock 5D
    bench = Benchmark("rosenbrock", dim=5)
    X = bench.sample_sobol(50)
    Y = bench.evaluate(X)

    # Injection d’outliers
    Xo, Yo = bench.inject_outliers(X, Y, n_outliers=5, mode="both")
    print("Original Y (min,max):", Y.min(), Y.max())
    print("With outliers Y (min,max):", Yo.min(), Yo.max())

 