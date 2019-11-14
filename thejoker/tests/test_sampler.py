# Standard library
import os

# Third-party
import astropy.units as u
from astropy.time import Time
import numpy as np
import pytest
from schwimmbad import SerialPool, MultiPool
from twobody import KeplerOrbit

# Package
from ..prior import JokerPrior
from ..data import RVData
from ..thejoker import TheJoker
from .test_prior import get_prior


def make_data(n_times=8, random_state=None, v1=None, K=None):
    if random_state is None:
        random_state = np.random.RandomState()
    rnd = random_state

    P = 51.8239 * u.day
    if K is None:
        K = 54.2473 * u.km/u.s
    v0 = 31.48502 * u.km/u.s
    EPOCH = Time('J2000')
    t = Time('J2000') + P * np.sort(rnd.uniform(0, 3., n_times))

    # binary system - random parameters
    orbit = KeplerOrbit(P=P, K=K, e=0.3, omega=0.283*u.radian,
                        M0=2.592*u.radian, t0=EPOCH,
                        i=90*u.deg, Omega=0*u.deg)  # these don't matter

    rv = orbit.radial_velocity(t) + v0
    if v1 is not None:
        rv = rv + v1 * (t - EPOCH).jd * u.day

    err = np.full_like(rv.value, 0.5) * u.km/u.s
    data = RVData(t, rv, rv_err=err)

    return data, orbit


@pytest.mark.parametrize('case', range(get_prior()))
def test_init(case):
    prior, _ = get_prior(case)

    # Try various initializations
    TheJoker(prior)

    with pytest.raises(TypeError):
        TheJoker('jsdfkj')

    # Pools:
    with SerialPool() as pool:
        TheJoker(prior, pool=pool)

    # fail when pool is invalid:
    with pytest.raises(TypeError):
        TheJoker(prior, pool='sdfks')

    # Random state:
    rnd = np.random.RandomState(42)
    TheJoker(prior, random_state=rnd)

    # fail when random state is invalid:
    with pytest.raises(TypeError):
        TheJoker(prior, random_state='sdfks')

    # tempfile location:
    joker = TheJoker(prior, tempfile_path='/tmp/joker')
    assert os.path.exists(joker.tempfile_path)


@pytest.mark.parametrize('case', range(get_prior()))
def test_marginal_ln_likelihood(tmpdir, case):
    prior, _ = get_prior(case)

    data, _ = make_data()
    prior_samples = prior.sample(size=100)
    joker = TheJoker(prior)

    # pass JokerSamples instance
    ll = joker.marginal_ln_likelihood(data, prior_samples)
    assert len(ll) == len(prior_samples)

    # save prior samples to a file and pass that instead
    filename = str(tmpdir / 'samples.hdf5')
    prior_samples.write(filename, overwrite=True)

    ll = joker.marginal_ln_likelihood(data, filename)
    assert len(ll) == len(prior_samples)

    # make sure batches work:
    ll = joker.marginal_ln_likelihood(data, filename,
                                      n_batches=10)
    assert len(ll) == len(prior_samples)

    # NOTE: this makes it so I can't parallelize tests, I think
    with MultiPool(processes=2) as pool:
        joker = TheJoker(prior, pool=pool)
        ll = joker.marginal_ln_likelihood(data, filename)
    assert len(ll) == len(prior_samples)


@pytest.mark.parametrize('prior', [
    JokerPrior.default(P_min=5*u.day, P_max=500*u.day,
                       sigma_K0=25*u.km/u.s, sigma_v=100*u.km/u.s),
    JokerPrior.default(P_min=5*u.day, P_max=500*u.day,
                       sigma_K0=25*u.km/u.s, poly_trend=2,
                       sigma_v=[100*u.km/u.s, 0.5*u.km/u.s/u.day])
])
def test_rejection_sample(tmpdir, prior):
    data, orbit = make_data()
    flat_data, orbit = make_data(K=0.1*u.m/u.s)

    prior_samples = prior.sample(size=16384)
    filename = str(tmpdir / f'samples.hdf5')
    prior_samples.write(filename, overwrite=True)

    joker = TheJoker(prior)

    for _samples in [prior_samples, filename]:
        # pass JokerSamples instance, process all samples:
        samples = joker.rejection_sample(data, _samples)
        assert len(samples) > 0
        assert len(samples) < 10  # HACK: this should generally be true...

        samples = joker.rejection_sample(flat_data, _samples)
        assert len(samples) > 10  # HACK: this should generally be true...


@pytest.mark.skip(reason="TODO: need to re-implement this!")
def test_mcmc_continue():
    rnd = np.random.RandomState(42)

    # First, try just running rejection_sample()
    data = self.data['binary']
    joker = TheJoker(self.joker_params['binary'], random_state=rnd)

    samples = joker.rejection_sample(data, n_prior_samples=16384)
    joker.mcmc_sample(data, samples, n_steps=8, n_burn=8,
                        n_walkers=128, return_sampler=False)
    joker.mcmc_sample(data, samples, n_steps=8, n_burn=8, n_walkers=128,
                        return_sampler=True)

    # Fancy K prior:
    data = self.data['binary_Kprior']
    joker = TheJoker(self.joker_params['binary_Kprior'], random_state=rnd)

    samples = joker.rejection_sample(data, n_prior_samples=16384)
    joker.mcmc_sample(data, samples, n_steps=8, n_burn=8,
                        n_walkers=128, return_sampler=False)
