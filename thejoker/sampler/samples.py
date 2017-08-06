# Standard library
from collections import OrderedDict

# Third-party
import numpy as np

# Package
from ..utils import quantity_to_hdf5, quantity_from_hdf5

__all__ = ['JokerSamples']

class JokerSamples(OrderedDict):

    _valid_keys = ['P', 'phi0', 'ecc', 'omega', 'jitter', 'K'] + \
                  ['v{}'.format(i) for i in range(10)]

    def __init__(self, **kwargs):
        """ """

        kw = kwargs.copy()

        self._n_samples = None
        for key,val in kw.items():
            self._validate_key(key)
            kw[key] = self._validate_val(val)

            if self._n_samples is None:
                self._n_samples = len(val)

        super(JokerSamples, self).__init__(**kw)

    def _validate_key(self, key):
        if key not in self._valid_keys:
            raise ValueError("Invalid key '{}'.".format(key))

    def _validate_val(self, val):
        if self._n_samples is not None and len(val) != self._n_samples:
            raise ValueError("Length of new samples must match those already stored!"
                             "({}, expected {})".format(len(val), self._n_samples))

        return np.atleast_1d(val)

    def __getitem__(self, slc):
        if isinstance(slc, str):
            return super(JokerSamples, self).__getitem__(slc)

        else:
            new = self.__class__()
            for k in self.keys():
                new[k] = self[k][slc]
            return new

    def __setitem__(self, key, val):
        self._validate_key(key)
        val = self._validate_val(val)

        if self._n_samples is None:
            self._n_samples = len(val)

        super(JokerSamples, self).__setitem__(key, val)

    @property
    def n_samples(self):
        if self._n_samples is None:
            raise ValueError("No samples stored!")
        return self._n_samples

    def __len__(self):
        return self.n_samples

    def __str__(self):
        return "<JokerSamples: n={}>".format()

    @classmethod
    def from_hdf5(cls, f, n=None):
        """
        Parameters
        ----------
        f : :class:`h5py.File`, :class:`h5py.Group`
        n : int (optional)
            The number of samples to load.
        """
        samples = cls()
        for key in f.keys():
            samples[key] = quantity_from_hdf5(f, key, n=n)

        return samples

    def to_hdf5(self, f):
        """
        Parameters
        ----------
        f : :class:`h5py.File`, :class:`h5py.Group`
        """

        for key in self.keys():
            quantity_to_hdf5(f, key, self[key])