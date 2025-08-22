import numpy as np

from scipy.stats import qmc
 
class Benchmark:

    def __init__(self, name: str, dim: int):

        """

        Initialise le benchmark choisi.

        name : str : 'rosenbrock' ou 'kursawe'

        dim : int : dimension du problème

        """

        self.name = name.lower()

        self.dim = dim

        self.bounds = self._set_bounds()

    def _set_bounds(self):

        """ Définit les bornes usuelles pour le benchmark. """

        if self.name == "rosenbrock":

            # Domaine classique : [-5, 10]^d

            return np.array([[-5.0, 10.0]] * self.dim)

        elif self.name == "kursawe":

            # Domaine standard : [-5, 5]^3

            if self.dim != 3:

                raise ValueError("Kursawe est défini en 3 dimensions")

            return np.array([[-5.0, 5.0]] * self.dim)

        else:

            raise ValueError(f"Benchmark {self.name} non reconnu.")

    def sample_sobol(self, n_points: int, scramble: bool = True):

        """ Génère un plan Sobol. """

        sampler = qmc.Sobol(d=self.dim, scramble=scramble)

        sample = sampler.random(n_points)

        return qmc.scale(sample, self.bounds[:,0], self.bounds[:,1])

    def sample_halton(self, n_points: int, scramble: bool = True):

        """ Génère un plan Halton. """

        sampler = qmc.Halton(d=self.dim, scramble=scramble)

        sample = sampler.random(n_points)

        return qmc.scale(sample, self.bounds[:,0], self.bounds[:,1])

    def sample_lhs(self, n_points: int, criterion: str = "maximin", iterations: int = 1000):

        """ Génère un plan Latin Hypercube Sampling. """

        sampler = qmc.LatinHypercube(d=self.dim, optimization=criterion)

        sample = sampler.random(n_points)

        return qmc.scale(sample, self.bounds[:,0], self.bounds[:,1])
 
# ------------------------

# Exemple d’utilisation :

if __name__ == "__main__":

    # Rosenbrock 2D

    bench_rosen = Benchmark("rosenbrock", dim=2)

    X_rosen = bench_rosen.sample_sobol(128)

    print("Sobol Rosenbrock 2D:\n", X_rosen[:5])

    # Kursawe 3D

    bench_kursawe = Benchmark("kursawe", dim=3)

    X_kursawe = bench_kursawe.sample_lhs(100)

    print("LHS Kursawe 3D:\n", X_kursawe[:5])

 