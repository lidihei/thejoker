# Third-party
import astropy.units as u
import numpy as np
import pymc3 as pm
from pymc3.distributions import draw_values
import theano.tensor as tt
from exoplanet.distributions.eccentricity import kipping13
import exoplanet.units as xu

# Project
from .samples import JokerSamples
from .prior_helpers import (UniformLog, FixedCompanionMass,
                            _validate_polytrend, _get_nonlinear_equiv_units,
                            _get_linear_equiv_units)

__all__ = ['JokerPrior']


def _validate_model(model):
    # validate input model
    if model is None:
        try:
            # check to see if we are in a context
            model = pm.modelcontext(None)
        except TypeError:  # we are not!
            # if no model is specified, create one and hold onto it
            model = pm.Model()

    if not isinstance(model, pm.Model):
        raise TypeError("Input model must be a pymc3.Model instance, not "
                        "a {}".format(type(model)))

    return model


@u.quantity_input(P_min=u.day, P_max=u.day)
def default_nonlinear_prior(P_min=None, P_max=None, s=None,
                            model=None, pars=None):
    r"""
    Retrieve pymc3 variables that specify the default prior on the nonlinear
    parameters of The Joker. See docstring of `JokerPrior.default()` for more
    information.

    The nonlinear parameters an default prior forms are:

    * ``P``, period: :math:`p(P) \propto 1/P`, over the domain
      :math:`(P_{\rm min}, P_{\rm max})`
    * ``e``, eccentricity: the short-period form from Kipping (2013)
    * ``M0``, phase: uniform over the domain :math:`(0, 2\pi)`
    * ``omega``, argument of pericenter: uniform over the domain
      :math:`(0, 2\pi)`
    * ``s``, additional extra variance added in quadrature to data
      uncertainties: delta-function at 0

    Parameters
    ----------
    P_min : `~astropy.units.Quantity` [time]
    P_max : `~astropy.units.Quantity` [time]
    s : `~pm.model.TensorVariable`, ~astropy.units.Quantity` [speed]
    model : `pymc3.Model`
        This is either required, or this function must be called within a pymc3
        model context.
    """
    model = pm.modelcontext(model)

    if pars is None:
        pars = dict()

    if s is None:
        s = 0 * u.m/u.s

    if isinstance(s, pm.model.TensorVariable):
        pars['s'] = pars.get('s', s)
    else:
        if not hasattr(s, 'unit') or not s.unit.is_equivalent(u.km/u.s):
            raise u.UnitsError("Invalid unit for s: must be equivalent to km/s")

    # dictionary of parameters to return
    out_pars = dict()

    with model:
        # Set up the default priors for parameters with defaults

        # Note: we have to do it this way (as opposed to with .get(..., default)
        # because this can only get executed if the param is not already
        # defined, otherwise variables are defined twice in the model
        if 'e' not in pars:
            out_pars['e'] = xu.with_unit(kipping13('e'),
                                         u.one)

        if 'omega' not in pars:
            out_pars['omega'] = xu.with_unit(pm.Uniform('omega',
                                                        lower=0,
                                                        upper=2*np.pi),
                                             u.radian)

        if 'M0' not in pars:
            out_pars['M0'] = xu.with_unit(pm.Uniform('M0',
                                                     lower=0,
                                                     upper=2*np.pi),
                                          u.radian)

        if 's' not in pars:
            out_pars['s'] = xu.with_unit(pm.Constant('s', s.value),
                                         s.unit)

        if 'P' not in pars:
            if P_min is None or P_max is None:
                raise ValueError("If you are using the default period prior, "
                                 "you must pass in both P_min and P_max to set "
                                 "the period prior domain.")
            out_pars['P'] = xu.with_unit(UniformLog('P',
                                                    P_min.value,
                                                    P_max.to_value(P_min.unit)),
                                         P_min.unit)

    for k in pars.keys():
        out_pars[k] = pars[k]

    return out_pars


def _validate_sigma_v(sigma_v, poly_trend, v_names):
    if isinstance(sigma_v, u.Quantity):
        if not sigma_v.isscalar:
            raise ValueError("You must pass in a scalar value for sigma_v if "
                             "passing in a single quantity.")
        sigma_v = {'v0': sigma_v}

    if hasattr(sigma_v, 'keys'):
        for name in v_names:
            if name not in sigma_v.keys():
                raise ValueError("If specifying the standard-deviations of "
                                 "the polynomial trend parameter prior, you "
                                 "must pass in values for all parameter names."
                                 "Expected keys: {}, received: {}"
                                 .format(v_names, sigma_v.keys()))
        return sigma_v

    try:
        if len(sigma_v) != poly_trend:
            raise ValueError("You must pass in a single sigma value for "
                             "each velocity trend parameter: You passed in "
                             "{} values, but poly_trend={}"
                             .format(len(sigma_v), poly_trend))
        sigma_v = {name: val for name, val in zip(v_names, sigma_v)}

    except TypeError:
        raise TypeError("Invalid input for velocity trend prior sigma "
                        "values. This must either be a scalar Quantity (if "
                        "poly_trend=1) or an iterable of Quantity objects "
                        "(if poly_trend>1)")

    return sigma_v


@u.quantity_input(sigma_K0=u.km/u.s, P0=u.day)
def default_linear_prior(sigma_K0=None, P0=None, sigma_v=None,
                         poly_trend=1, model=None, pars=None):
    r"""
    Retrieve pymc3 variables that specify the default prior on the linear
    parameters of The Joker. See docstring of `JokerPrior.default()` for more
    information.

    The linear parameters an default prior forms are:

    * ``K``, velocity semi-amplitude: Normal distribution, but with a variance
      that scales with period and eccentricity.
    * ``v0``, ``v1``, etc. polynomial velocity trend parameters: Independent
      Normal distributions.

    Parameters
    ----------
    sigma_K0 : `~astropy.units.Quantity` [speed]
    P0 : `~astropy.units.Quantity` [time]
    sigma_v : iterable of `~astropy.units.Quantity`
    model : `pymc3.Model`
        This is either required, or this function must be called within a pymc3
        model context.
    """
    model = pm.modelcontext(model)

    if pars is None:
        pars = dict()

    # dictionary of parameters to return
    out_pars = dict()

    # set up poly. trend names:
    poly_trend, v_names = _validate_polytrend(poly_trend)

    # get period/ecc from dict of nonlinear parameters
    P = model.named_vars.get('P', None)
    e = model.named_vars.get('e', None)
    if P is None or e is None:
        raise ValueError("Period P and eccentricity e must both be defined as "
                         "nonlinear parameters on the model.")

    if v_names and 'v0' not in pars:
        sigma_v = _validate_sigma_v(sigma_v, poly_trend, v_names)

    with model:
        if 'K' not in pars:
            if sigma_K0 is None or P0 is None:
                raise ValueError("If using the default prior form on K, you "
                                 "must pass in a variance scale (sigma_K0) "
                                 "and a reference period (P0)")

            # Default prior on semi-amplitude: scales with period and
            # eccentricity such that it is flat with companion mass
            v_unit = sigma_K0.unit
            out_pars['K'] = xu.with_unit(FixedCompanionMass('K', P=P, e=e,
                                                            sigma_K0=sigma_K0,
                                                            P0=P0),
                                         v_unit)
        else:
            v_unit = getattr(pars['K'], xu.UNIT_ATTR_NAME, u.one)

        for i, name in enumerate(v_names):
            if name not in pars:
                # Default priors are independent gaussians
                # TODO: FIXME: make mean, mu_v, customizable
                out_pars[name] = xu.with_unit(
                    pm.Normal(name, 0.,
                              sigma_v[name].value),
                    sigma_v[name].unit)

    for k in pars.keys():
        out_pars[k] = pars[k]

    return out_pars


class JokerPrior:

    def __init__(self, pars=None, poly_trend=1, v0_offsets=None, model=None):
        """
        This class controls the prior probability distributions for the
        parameters used in The Joker.

        This initializer is meant to be flexible, allowing you to specify the
        prior distributions on the linear and nonlinear parameters used in The
        Joker. However, for many use cases, you may want to just use the
        default prior: To initialize this object using the default prior, see
        the alternate initializer `JokerPrior.default()`.

        Parameters
        ----------
        pars : dict, list (optional)
            Either a list of pymc3 variables, or a dictionary of variables with
            keys set to the variable names. If any of these variables are
            defined as deterministic transforms from other variables, see the
            next parameter below.
        poly_trend : int (optional)
            Specifies the number of coefficients in an additional polynomial
            velocity trend, meant to capture long-term trends in the data. The
            default here is ``polytrend=1``, meaning one term: the (constant)
            systemtic velocity. For example, ``poly_trend=3`` will sample over
            parameters of a long-term quadratic velocity trend.
        v0_offsets : list (optional)
            A list of additional Gaussian parameters that set systematic offsets
            of subsets of the data. TODO: link to tutorial here
        model : `pymc3.Model`
            This is either required, or this function must be called within a
            pymc3 model context.

        """

        self.model = _validate_model(model)

        # Parse and clean up the input pars
        if pars is None:
            pars = dict()
            pars.update(model.named_vars)

        elif isinstance(pars, tt.TensorVariable):  # a single variable
            # Note: this has to go before the next clause because TensorVariable
            # instances are iterable...
            pars = {pars.name: pars}

        else:
            try:
                pars = dict(pars)  # try to coerce to a dictionary
            except Exception:
                # if that fails, assume it is an iterable, like a list or tuple
                try:
                    pars = {p.name: p for p in pars}
                except Exception:
                    raise ValueError("Invalid input parameters: The input "
                                     "`pars` must either be a dictionary, "
                                     "list, or a single pymc3 variable, not a "
                                     "'{}'.".format(type(pars)))

        # Set the number of polynomial trend parameters
        self.poly_trend, self._v_trend_names = _validate_polytrend(poly_trend)

        # TODO: FIXME: enable support for this
        # TODO: support passing in v0_offsets as a dict with same keys as data
        if v0_offsets is None:
            v0_offsets = []

        try:
            v0_offsets = list(v0_offsets)
        except Exception:
            raise TypeError("Constant velocity offsets must be an iterable "
                            "of pymc3 variables that define the priors on "
                            "each offset term.")

        self._v0_offset_pars = {p.name: u.m/u.s for p in v0_offsets}
        self._n_offsets = len(v0_offsets)
        self.v0_offsets = v0_offsets
        pars.update({p.name: p for p in self.v0_offsets})

        # Store the names of the default parameters, used for validating input:
        # Note: these are *not* the units assumed internally by the code, but
        # are only used to validate that the units for each parameter are
        # equivalent to these
        self._nonlinear_pars = _get_nonlinear_equiv_units()
        self._linear_pars = _get_linear_equiv_units(self._v_trend_names)
        self._all_par_unit_equiv = {**self._nonlinear_pars,
                                    **self._linear_pars,
                                    **self._v0_offset_pars}

        # At this point, pars must be a dictionary: validate that all
        # parameters are specified and that they all have units
        for name in self.par_names:
            if name not in pars:
                raise ValueError(f"Missing prior for parameter '{name}': "
                                 "you must specify a prior distribution for "
                                 "all parameters.")

            if not hasattr(pars[name], xu.UNIT_ATTR_NAME):
                raise ValueError(f"Parameter '{name}' does not have associated "
                                 "units: Use exoplanet.units to specify units "
                                 "for your pymc3 variables. See the "
                                 "documentation for examples: thejoker.rtfd.io")

            equiv_unit = self._all_par_unit_equiv[name]
            if not getattr(pars[name],
                           xu.UNIT_ATTR_NAME).is_equivalent(equiv_unit):
                raise ValueError(f"Parameter '{name}' has an invalid unit: "
                                 f"The units for this parameter must be "
                                 f"transformable to '{equiv_unit}'")

        # Enforce that the priors on all linear parameters are Normal (or normal subclass)
        for name in (list(self._linear_pars.keys())
                     + list(self._v0_offset_pars.keys())):
            if not isinstance(pars[name].distribution, pm.Normal):
                raise ValueError("Priors on the linear parameters (K, v0, "
                                 "etc.) must be independent Normal "
                                 "distributions, not '{}'"
                                 .format(type(pars[name].distribution)))

        self.pars = pars

    @classmethod
    def default(cls, P_min, P_max, sigma_K0=None, P0=1*u.year, sigma_v=None,
                s=None, poly_trend=1, v0_offsets=None, model=None, pars=None):
        r"""
        An alternative initializer to set up the default prior for The Joker.

        The default prior is:

        .. math::

            p(P) \propto \frac{1}{P} \quad \elem (P_{\rm min}, P_{\rm max})
            p(e) = B(a_e, b_e)
            p(\omega) = \mathcal{U}(0, 2\pi)
            p(M_0) = \mathcal{U}(0, 2\pi)
            p(s) = \delta(s)
            p(K) \propto \mathcal{N}(K \,|\, \mu_K, \sigma_K)
            \sigma_K = \sigma_{K, 0} \, \left(\frac{P}{P_0}\right)^{-1/3} \, \left(1 - e^2\right)^{-1}

        and the priors on any polynomial trend parameters are assumed to be
        independent, univariate Normals.

        This prior has sensible choices for typical binary star or exoplanet
        use cases, but if you need more control over the prior distributions
        you might need to use the standard initializer (i.e.
        ``JokerPrior(...)```) and specify all parameter distributions manually.
        See `the documentation <http://thejoker.readthedocs.io>`_ for tutorials
        that demonstrate this functionality.

        Parameters
        ----------
        P_min : `~astropy.units.Quantity` [time]
            Minimum period for the default period prior.
        P_max : `~astropy.units.Quantity` [time]
            Maximum period for the default period prior.
        sigma_K0 : `~astropy.units.Quantity` [speed]
            The scale factor, :math:`\sigma_{K, 0}` in the equation above that
            sets the scale of the semi-amplitude prior at the reference period,
            ``P0``.
        P0 : `~astropy.units.Quantity` [time]
            The reference period, :math:`P_0`, used in the prior on velocity
            semi-amplitude (see equation above).
        sigma_v : `~astropy.units.Quantity` (or iterable of)
            The standard deviations of the velocity trend priors.
        s : `~astropy.units.Quantity` [speed]
            The jitter value, assuming it is constant.
        poly_trend : int (optional)
            Specifies the number of coefficients in an additional polynomial
            velocity trend, meant to capture long-term trends in the data. The
            default here is ``polytrend=1``, meaning one term: the (constant)
            systemtic velocity. For example, ``poly_trend=3`` will sample over
            parameters of a long-term quadratic velocity trend.
        v0_offsets : list (optional)
            A list of additional Gaussian parameters that set systematic offsets
            of subsets of the data. TODO: link to tutorial here
        model : `pymc3.Model` (optional)
            If not specified, this will create a model instance and store it on the prior object.
        pars : dict, list (optional)
            Either a list of pymc3 variables, or a dictionary of variables with
            keys set to the variable names. If any of these variables are
            defined as deterministic transforms from other variables, see the
            next parameter below.
        """

        model = _validate_model(model)

        nl_pars = default_nonlinear_prior(P_min, P_max, s=s,
                                          model=model, pars=pars)
        l_pars = default_linear_prior(sigma_K0=sigma_K0, P0=P0, sigma_v=sigma_v,
                                      poly_trend=poly_trend, model=model,
                                      pars=pars)

        pars = {**nl_pars, **l_pars}
        obj = cls(pars=pars, model=model, poly_trend=poly_trend,
                  v0_offsets=v0_offsets)

        return obj

    @property
    def par_names(self):
        return (list(self._nonlinear_pars.keys())
                + list(self._linear_pars.keys())
                + [off.name for off in self.v0_offsets])

    @property
    def par_units(self):
        return {p.name: getattr(p, xu.UNIT_ATTR_NAME, u.one) for p in self.pars}

    def __repr__(self):
        return f'<JokerPrior [{", ".join(self.par_names)}]>'

    def __str__(self):
        return ", ".join(self.par_names)

    def sample(self, size=1, generate_linear=False, return_logprobs=False):
        """
        Generate random samples from the prior.

        .. note::

            Right now, generating samples with the prior values is slow (i.e.
            with ``return_logprobs=True``) because of pymc3 issues (see
            discussion here: https://discourse.pymc.io/t/draw-values-speed-scaling-with-transformed-variables/4076).
            This will hopefully be resolved in the future...

        Parameters
        ----------
        size : int (optional)
            The number of samples to generate.
        generate_linear : bool (optional)
            Also generate samples in the linear parameters.
        return_logprobs : bool (optional)
            Generate the log-prior probability at the position of each sample.

        Returns
        -------
        samples : `thejoker.Jokersamples`
            The random samples.

        """
        sub_pars = {k: p for k, p in self.pars.items()
                    if k in self._nonlinear_pars
                    or ((k in self._linear_pars or k in self._v0_offset_pars)
                        and generate_linear)}

        if generate_linear:
            # TODO: we could warn that this is usually slow because of pymc3?
            par_names = self.par_names
        else:
            par_names = list(self._nonlinear_pars.keys())

        pars_list = list(sub_pars.values())
        npars = len(pars_list)

        log_prior = []
        if return_logprobs:
            # Note: This is really slow! Waiting for upstream fixes to pymc3...

            # Add deterministic variables to track the value of the prior at
            # each sample generated:
            with self.model:
                for par in pars_list:
                    logp_name = f'{par.name}_log_prior'
                    dist = par.distribution.logp(par)

                    if logp_name in self.model.named_vars.keys():
                        logp_var = self.model.named_vars[logp_name]
                    else:
                        # doesn't exist in the model yet, so add it
                        logp_var = pm.Deterministic(logp_name, dist)

                    log_prior.append(logp_var)

        samples_values = draw_values(pars_list + log_prior, size=size)
        raw_samples = {p.name: samples
                       for p, samples in zip(pars_list,
                                             samples_values[:npars])}

        # Apply units if they are specified:
        prior_samples = JokerSamples(prior=self)
        for name in par_names:
            p = sub_pars[name]
            unit = getattr(p, xu.UNIT_ATTR_NAME, u.one)

            if p.name not in prior_samples._valid_units.keys():
                continue

            prior_samples[p.name] = np.atleast_1d(raw_samples[p.name]) * unit

        if return_logprobs:
            log_prior = {p.name: vals
                         for p, vals in zip(pars_list, samples_values[npars:])}
            prior_samples['ln_prior'] = np.sum([v for v in log_prior.values()],
                                               axis=0)

        # TODO: right now, elsewhere, we assume the log_prior is a single value
        # for each sample (i.e. the total prior value). In principle, we could
        # store all of the individual log-prior values (for each parameter),
        # like here:
        # log_prior = {k: np.atleast_1d(v)
        #              for k, v in log_prior.items()}
        # log_prior = Table(log_prior)[par_names]

        return prior_samples
