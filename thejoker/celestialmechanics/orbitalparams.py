# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import h5py
import numpy as np
import six

# Project
from ..util import quantity_from_hdf5
from ..units import usys

class OrbitalParams(object):
    # Mapping from parameter name to physical type
    _name_phystype = {
        'P': 'time',
        'asini': 'length',
        'ecc': None,
        'omega': 'angle',
        'phi0': 'angle',
        'v0': 'speed'
    }

    # Latex labels for the parameters
    labels = ['P', r'$a\,\sin i$', '$e$', r'$\omega$', r'$\phi_0$', '$v_0$']

    @u.quantity_input(P=u.day, asini=u.R_sun, omega=u.degree, phi0=u.degree, v0=u.km/u.s)
    def __init__(self, P, asini, ecc, omega, phi0, v0):
        """
        """

        # parameters are stored internally without units (for speed) but can
        #   be accessed with units without the underscore prefix (e.g., .P vs ._P)
        self._P = np.atleast_1d(P).decompose(usys).value
        self._asini = np.atleast_1d(asini).decompose(usys).value
        self._ecc = np.atleast_1d(ecc)
        self._omega = np.atleast_1d(omega).decompose(usys).value
        self._phi0 = np.atleast_1d(phi0).decompose(usys).value
        self._v0 = np.atleast_1d(v0).decompose(usys).value

        # validate shape of inputs
        if self._P.ndim > 1:
            raise ValueError("Only ndim=1 arrays are supported!")

        for key in self._name_phystype.keys():
            if getattr(self, key).shape != self._P.shape:
                raise ValueError("All inputs must have the same length!")

    def __getattr__(self, name):
        # this is a crazy hack to automatically apply units to attributes
        #   named after each of the parameters
        if name not in self._name_phystype.keys():
            raise AttributeError("Invalid attribute name '{}'".format(name))

        # get unitless representation
        val = getattr(self, '_{}'.format(name))

        if self._name_phystype[name] is None:
            return val
        else:
            return val * usys[self._name_phystype[name]]

    @classmethod
    def from_hdf5(cls, f):
        kwargs = dict()
        if isinstance(f, six.string_types):
            with h5py.File(f, 'r') as g:
                for key in cls._name_phystype.keys():
                    kwargs[key] = quantity_from_hdf5(g, key)

        else:
            for key in cls._name_phystype.keys():
                kwargs[key] = quantity_from_hdf5(f, key)

        return cls(**kwargs)
