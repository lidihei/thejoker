"""
Could have something like:

    # remember: right now, priors on these terms are assumed to be broad and Gaussian
    pvt = PolynomialVelocityTrend(n_terms=2) # linear
    pvt.data_mask = lambda d: d._t_bmjd < 55562.24 # only apply to data before a date

    # todo: marginalization then has to happen piece-wise?!

Future:

- Maybe make NonlinearParameter classes so that even nonlinear parameters can be extended?
- Parameter class has a .sample_prior() method, .evaluate_prior()??
- Some parameters are just required, need to be packed -- multiproc_helpers
    functions just need to unpack/pack smarter?

"""

# Third-party
from astropy.constants import G
import astropy.units as u
import h5py
import numpy as np
import six

# Project
from ..util import quantity_from_hdf5, quantity_to_hdf5

__all__ = ['JokerParams']

class JokerParams(object):
    """

    Parameters
    ----------
    P : `~astropy.units.Quantity` [time]
        Period.
    K : `~astropy.units.Quantity` [speed]
        Velocity semi-amplitude.
    ecc : numeric, array_like
        Eccentricity.
    omega : `~astropy.units.Quantity` [angle]
        Argument of pericenter.
    phi0 : `~astropy.units.Quantity` [angle]
        TODO:
    v0 : `~astropy.units.Quantity` [speed]
        Systemic (Barycenter) velocity of the system.
    jitter : `~astropy.units.Quantity` [time] (optional)
        Additional noise in the RV signal.
    """
    @u.quantity_input(P=u.day,
                      K=u.km/u.s,
                      omega=u.radian,
                      phi0=u.radian,
                      v0=u.km/u.s)
    def __init__(self, P, K, ecc, omega, phi0, v0, jitter=None):

        # parameters are stored internally without units (for speed) but can
        #   be accessed with units without the underscore prefix (e.g., .P vs ._P)
        for name,unit in self._name_to_unit.items():
            if unit is not None:
                if unit is u.one:
                    setattr(self, "_{}".format(name), np.array(np.atleast_1d(eval(name))))
                else:
                    setattr(self, "_{}".format(name), np.atleast_1d(eval(name)).to(unit).value)
            else:
                setattr(self, "_{}".format(name), np.atleast_1d(eval(name)))

        # validate shape of inputs
        if self._P.ndim > 1:
            raise ValueError("Only ndim=1 arrays are supported!")

        for key in self._name_to_unit.keys():
            if getattr(self, key).shape != self._P.shape:
                raise ValueError("All inputs must have the same length!")

    def __getattr__(self, name):
        # this is a crazy hack to automatically apply units to attributes
        #   named after each of the parameters
        if name not in self._name_to_unit.keys():
            raise AttributeError("Invalid attribute name '{}'".format(name))

        # get unitless representation
        val = getattr(self, '_{}'.format(name))

        return val * self._name_to_unit[name]

    def __len__(self):
        return len(self._P)

    def __copy__(self):
        kw = dict()
        for key in self._name_to_unit.keys():
            kw[key] = getattr(self, key).copy()
        return self.__class__(**kw)

    def __getitem__(self, slicey):
        cpy = self.copy()
        for key in self._name_to_unit.keys():
            slice_val = getattr(self, "_{}".format(key))[slicey]
            setattr(cpy, "_{}".format(key), slice_val)
        return cpy

    @classmethod
    def get_labels(cls, units=None):
        _u = dict()
        if units is None:
            _u = cls._name_to_unit

        else:
            for k,unit in cls._name_to_unit.items():
                if k in units:
                    _u[k] = units[k]
                else:
                    _u[k] = unit

        _labels = [
            r'$\ln (P/1\,${}$)$'.format(_u['P'].long_names[0]),
            '$e$',
            r'$\omega$ [{}]'.format(_u['omega']),
            r'$\phi_0$ [{}]'.format(_u['omega']),
            r'$\ln (s/1\,${}$)$'.format(_u['jitter'].to_string(format='latex_inline')),
            r'$K$ [{}]'.format(_u['K'].to_string(format='latex_inline')),
            '$v_0$ [{}]'.format(_u['v0'].to_string(format='latex_inline'))
        ]

        return _labels

    @classmethod
    def from_hdf5(cls, f):
        kwargs = dict()
        if isinstance(f, six.string_types):
            with h5py.File(f, 'r') as g:
                for key in cls._name_to_unit.keys():
                    kwargs[key] = quantity_from_hdf5(g, key)

        else:
            for key in cls._name_to_unit.keys():
                kwargs[key] = quantity_from_hdf5(f, key)

        return cls(**kwargs)

    def to_hdf5(self, f):
        if isinstance(f, six.string_types):
            with h5py.File(f, 'a') as g:
                for key in self._name_to_unit.keys():
                    quantity_to_hdf5(g, key, getattr(self, key))

        else:
            for key in self._name_to_unit.keys():
                quantity_to_hdf5(f, key, getattr(self, key))

    def pack(self, units=None, plot_transform=False):
        """
        Pack the orbital parameters into a single array structure
        without associated units. The components will have units taken
        from the unit system defined in `thejoker.units.usys`.

        Parameters
        ----------
        units : dict (optional)
        plot_transform : bool (optional)

        Returns
        -------
        pars : `numpy.ndarray`
            A single 2D array containing the parameter values with no
            units. Will have shape ``(n,6)``.

        """
        if units is None:
            all_samples = np.vstack([getattr(self, "_{}".format(key))
                                     for key in self._name_to_unit.keys()]).T

        else:
            all_samples = np.vstack([getattr(self, format(key)).to(units[key]).value
                                     for key in self._name_to_unit.keys()]).T

        if plot_transform:
            # ln P in plots:
            idx = list(self._name_to_unit.keys()).index('P')
            all_samples[:,idx] = np.log(all_samples[:,idx])

            # ln s in plots:
            idx = list(self._name_to_unit.keys()).index('jitter')
            all_samples[:,idx] = np.log(all_samples[:,idx])

        return all_samples

    @classmethod
    def unpack(cls, pars):
        """
        Unpack a 2D array structure containing the orbital parameters
        without associated units. Should have shape ``(n,6)`` where ``n``
        is the number of parameters.

        Returns
        -------
        p : `~thejoker.celestialmechanics.OrbitalParams`

        """
        kw = dict()
        par_arr = np.atleast_2d(pars).T
        for i,key in enumerate(cls._name_to_unit.keys()):
            kw[key] = par_arr[i] * cls._name_to_unit[key]

        return cls(**kw)

    def copy(self):
        return self.__copy__()

    def rv_orbit(self, index=None):
        """
        Get a `~thejoker.celestialmechanics.SimulatedRVOrbit` instance
        for the orbital parameters with index ``i``.

        Parameters
        ----------
        index : int (optional)

        Returns
        -------
        orbit : `~thejoker.celestialmechanics.SimulatedRVOrbit`
        """
        from .celestialmechanics_class import SimulatedRVOrbit

        if index is None and len(self._P) == 1: # OK
            index = 0

        elif index is None and len(self._P) > 1:
            raise IndexError("You must specify the index of the set of paramters to get an "
                             "orbit for!")

        i = index
        return SimulatedRVOrbit(self[i])

    # Computed Quantities
    @property
    def asini(self):
        return (self.K/(2*np.pi) * (self.P * np.sqrt(1-self.ecc**2))).to(default_units['asini'])

    @property
    def mf(self):
        mf = self.P * self.K**3 / (2*np.pi*G) * (1 - self.ecc**2)**(3/2.)
        return mf.to(default_units['mf'])

    @staticmethod
    def mf_asini_ecc_to_P_K(mf, asini, ecc):
        P = 2*np.pi * asini**(3./2) / np.sqrt(G * mf)
        K = 2*np.pi * asini / (P * np.sqrt(1-ecc**2))
        return P.to(default_units['P']), K.to(default_units['K'])