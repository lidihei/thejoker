from astropy.time import Time
import astropy.units as u
import numpy as np

from .src.fast_likelihood import CJokerSB2Helper
from .samples import JokerSamples
from .likelihood_helpers import get_trend_design_matrix
from .thejoker import TheJoker
from .prior import _validate_model
from .samples_analysis import is_P_unimodal
from astropy.table import QTable
from thejoker.logging import logger
from .prior_helpers import validate_n_offsets, validate_poly_trend


__all__ = ['TheJokerSB2', 'JokerSB2Samples']


def validate_prepare_data_sb2(data, poly_trend, t_ref=None):
    """Internal function.

    Used to take an input ``RVData`` instance, or a list/dict of ``RVData``
    instances, and produce concatenated time, RV, and error arrays, along
    with a consistent t_ref.
    """
    from .data import RVData

    # If we've gotten here, data is dict-like:
    rv_unit = None
    t = []
    rv = []
    err = []
    ids = []
    for k in data.keys():
        d = data[k]

        if not isinstance(d, RVData):
            raise TypeError(f"All data must be specified as RVData instances: "
                            f"Object at key '{k}' is a '{type(d)}' instead.")

        if d._has_cov:
            raise NotImplementedError("We currently don't support "
                                      "multi-survey data when a full "
                                      "covariance matrix is specified. "
                                      "Raise an issue in adrn/thejoker if "
                                      "you want this functionality.")

        if rv_unit is None:
            rv_unit = d.rv.unit

        t.append(d.t.tcb.mjd)
        rv.append(d.rv.to_value(rv_unit))
        err.append(d.rv_err.to_value(rv_unit))
        ids.append([k] * len(d))

    t = np.concatenate(t)
    rv = np.concatenate(rv) * rv_unit
    err = np.concatenate(err) * rv_unit
    ids = np.concatenate(ids)

    all_data = RVData(t=Time(t, format='mjd', scale='tcb'),
                      rv=rv, rv_err=err,
                      t_ref=t_ref, sort=False)
    K_M = np.zeros((len(all_data), 2))
    K_M[ids == '1', 0] = 1.
    K_M[ids == '2', 1] = -1.
    trend_M = get_trend_design_matrix(all_data, ids=None, poly_trend=poly_trend)

    return all_data, ids, np.hstack((K_M, trend_M))


class JokerSB2Samples(JokerSamples):

    def __init__(self, samples=None, t_ref=None, poly_trend=None,
                 **kwargs):
        """
        A dictionary-like object for storing prior or posterior samples from
        The Joker, with some extra functionality.

        Parameters
        ----------
        samples : `~astropy.table.QTable`, table-like (optional)
            The samples data as an Astropy table object, or something
            convertable to an Astropy table (e.g., a dictionary with
            `~astropy.units.Quantity` object values). This is optional because
            the samples data can be added later by setting keys on the resulting
            instance.
        poly_trend : int (optional)
            Specifies the number of coefficients in an additional polynomial
            velocity trend, meant to capture long-term trends in the data. See
            the docstring for `thejoker.JokerPrior` for more details.
        t_ref : `astropy.time.Time`, numeric (optional)
            The reference time for the orbital parameters.
        **kwargs
            Additional keyword arguments are stored internally as metadata.
        """
        super().__init__(t_ref=t_ref, poly_trend=poly_trend, **kwargs)
        self._valid_units['K1'] = self._valid_units.pop('K')
        self._valid_units['K2'] = self._valid_units.get('K1')

        if samples is not None:
            _tbl = QTable(samples)
            for colname in _tbl.colnames:
                self[colname] = np.atleast_1d(_tbl[colname])

    def __repr__(self):
        return (f'<JokerSB2Samples [{", ".join(self.par_names)}] '
                f'({len(self)} samples)>')

    @property
    def primary(self):
        new_samples = JokerSamples(t_ref=self.t_ref, poly_trend=self.poly_trend)
        new_samples['K'] = self['K1']
        for name in new_samples._valid_units.keys():
            if name == 'K' or name not in self.tbl.colnames:
                continue

            else:
                new_samples[name] = self[name]
        return new_samples

    @property
    def secondary(self):
        new_samples = JokerSamples(t_ref=self.t_ref, poly_trend=self.poly_trend)
        new_samples['K'] = self['K2']
        for name in new_samples._valid_units.keys():
            if name == 'K' or name not in self.tbl.colnames:
                continue

            elif name == 'omega':
                new_samples[name] = self[name] - 180*u.deg

            else:
                new_samples[name] = self[name]
        return new_samples

    #def get_orbit(self, index=None, which='1', **kwargs):
    #    pass


class TheJokerSB2(TheJoker):
    _samples_cls = JokerSB2Samples

    def _make_joker_helper(self, data):
        assert len(data) == 2
        assert '1' in data.keys() and '2' in data.keys()
        all_data, ids, M = validate_prepare_data_sb2(
            data, self.prior.poly_trend, t_ref=data['1'].t_ref)
        self.ids = ids
        joker_helper = CJokerSB2Helper(all_data, self.prior, M)
        return joker_helper

    def setup_mcmc(self, data, joker_samples, model=None, custom_func=None):
        """
        Setup the model to run MCMC using pymc.

        Parameters
        ----------
        data : `~thejoker.RVData`
            The radial velocity data, or an iterable containing ``RVData``
            objects for each data source.
        joker_samples : `~thejoker.JokerSamples`
            If a single sample is passed in, this is packed into a pymc
            initialization dictionary and returned after setting up. If
            multiple samples are passed in, the median (along period) sample is
            taken and returned after setting up for MCMC.
        model : `pymc.Model`
            This is either required, or this function must be called within a
            pymc model context.
        custom_func : callable (optional)

        Returns
        -------
        mcmc_init : dict

        """
        import pymc as pm
        import pytensor.tensor as pt

        import thejoker.units as xu
        from thejoker._keplerian_orbit import KeplerianOrbit

        model = _validate_model(model)

        # Reduce data, strip units:
        data, ids, _ = validate_prepare_data_sb2(
            data, self.prior.poly_trend, t_ref=None
        )
        x = data._t_bmjd - data._t_ref_bmjd
        y = data.rv.value
        err = data.rv_err.to_value(data.rv.unit)

        # First, prepare the joker_samples:
        if not isinstance(joker_samples, JokerSamples):
            raise TypeError(
                "You must pass in a JokerSamples instance to the "
                "joker_samples argument."
            )

        if len(joker_samples) > 1:
            # check if unimodal in P, if not, warn
            if not is_P_unimodal(joker_samples, data):
                logger.warn("TODO: samples ain't unimodal")

            MAP_sample = joker_samples.median_period()

        else:
            MAP_sample = joker_samples

        mcmc_init = {}
        for name in self.prior.par_names:
            unit = getattr(self.prior.pars[name], xu.UNIT_ATTR_NAME)
            mcmc_init[name] = MAP_sample[name].to_value(unit)
        if custom_func is not None:
            mcmc_init = custom_func(mcmc_init, MAP_sample, model)
        mcmc_init = {k: np.squeeze(v) for k, v in mcmc_init.items()}

        p = self.prior.pars

        if "t_peri" not in model.named_vars:
            with model:
                pm.Deterministic("t_peri", p["P"] * p["M0"] / (2 * np.pi))

        if "obs" in model.named_vars:
            return mcmc_init

        with model:
            # Set up the orbit model for star1
            orbit1 = KeplerianOrbit(
                period=p["P"],
                ecc=p["e"],
                omega=p["omega"],
                t_periastron=model.named_vars["t_peri"],
            )

        # design matrix

        # design matrix
        M = get_trend_design_matrix(data, ids, self.prior.poly_trend)

        # deal with v0_offsets, trend here:
        _, offset_names = validate_n_offsets(self.prior.n_offsets)
        _, vtrend_names = validate_poly_trend(self.prior.poly_trend)

        with model:
            v_pars = (
                [p["v0"]]
                + [p[name] for name in offset_names]
                + [p[name] for name in vtrend_names[1:]]
            )  # skip v0
            v_trend_vec = pt.stack(v_pars, axis=0)
            trend = pt.dot(M, v_trend_vec)

            rv_model = orbit.get_radial_velocity(x, K=p["K"]) + trend
            pm.Deterministic("model_rv", rv_model)

            err = pt.sqrt(err**2 + p["s"] ** 2)
            pm.Normal("obs", mu=rv_model, sigma=err, observed=y)

            pm.Deterministic("logp", model.logp())

            dist = pm.Normal.dist(model.model_rv, data.rv_err.value)
            lnlike = pm.Deterministic(
                "ln_likelihood", pm.logp(dist, data.rv.value).sum(axis=-1)
            )

            pm.Deterministic("ln_prior", model.logp() - lnlike)

        return mcmc_init
