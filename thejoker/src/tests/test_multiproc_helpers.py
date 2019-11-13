# Third-party
import astropy.units as u
import h5py
import numpy as np
import schwimmbad

# Package
# from ..multiproc_helpers import (get_good_sample_indices, compute_likelihoods,
#                                  sample_indices_to_full_samples, chunk_tasks)
from .helpers import FakeData


class TestMultiproc(object):

    def setup(self):
        d = FakeData()
        self.data = d.datasets
        self.joker_params = d.params
        self.truths = d.truths

    def test_multiproc_helpers(self, tmpdir):
        prior_samples_file = str(tmpdir.join('prior-samples.h5'))
        pool = schwimmbad.SerialPool()

        data = self.data['circ_binary']
        joker_params = self.joker_params['circ_binary']
        truth = self.truths['circ_binary']
        nlp = self.truths_to_nlp(truth)

        # write some nonsense out to the prior file
        n = 8192
        P = np.random.uniform(nlp[0]-2., nlp[0]+2., n)
        M0 = np.random.uniform(0, 2*np.pi, n)
        ecc = np.zeros(n)
        omega = np.zeros(n)
        jitter = np.zeros(n)
        samples = np.vstack((P,M0,ecc,omega,jitter)).T

        # TODO: use save_prior_samples here

        with h5py.File(prior_samples_file) as f:
            f['samples'] = samples

        lls = compute_likelihoods(n, prior_samples_file, 0, data,
                                  joker_params, pool)
        idx = get_good_sample_indices(lls)
        assert len(idx) >= 1

        lls = compute_likelihoods(n, prior_samples_file, 0, data,
                                  joker_params, pool, n_batches=13)
        idx = get_good_sample_indices(lls)
        assert len(idx) >= 1

        max_n_samples = 100
        full_samples = sample_indices_to_full_samples(idx, prior_samples_file,
                                                      data, joker_params,
                                                      max_n_samples, pool)
        print(full_samples)
