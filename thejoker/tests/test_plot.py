# Third-party
import astropy.units as u
import numpy as np
import pytest

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Package
from ..prior import JokerPrior
from ..plot import plot_rv_curves
from .test_sampler import make_data


@pytest.mark.skipif(not HAS_MPL, reason='matplotlib not installed')
@pytest.mark.parametrize('prior', [
    JokerPrior.default(10*u.day, 20*u.day,
                       25*u.km/u.s, sigma_v=100*u.km/u.s),
    JokerPrior.default(10*u.day, 20*u.day,
                       25*u.km/u.s, poly_trend=2,
                       sigma_v=[100*u.km/u.s, 0.2*u.km/u.s/u.day])
])
def test_plot_rv_curves(prior):

    data, _ = make_data()
    samples = prior.sample(100, generate_linear=True)

    t_grid = np.random.uniform(56000, 56500, 1024)
    t_grid.sort()

    plot_rv_curves(samples, t_grid)
    plot_rv_curves(samples, data=data)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plot_rv_curves(samples, t_grid, ax=ax)
