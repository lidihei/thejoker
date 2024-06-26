# Third-party
import astropy.units as u
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from astropy.utils.decorators import deprecated_renamed_argument

import thejoker.units as xu

# Project
from .logging import logger
from .prior_helpers import (
    get_linear_equiv_units,
    get_nonlinear_equiv_units,
    get_v0_offsets_equiv_units,
    validate_poly_trend,
    validate_sigma_v,
)

__all__ = ["JokerPrior"]


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
        msg = f"Input model must be a pymc.Model instance, not a {type(model)}"
        raise TypeError(msg)

    return model


class JokerPrior:
    _sb2 = False

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
            Either a list of pymc variables, or a dictionary of variables with
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
        model : `pymc.Model`
            This is either required, or this function must be called within a
            pymc model context.

        """
        self.model = _validate_model(model)

        # Parse and clean up the input pars
        if pars is None:
            pars = {}
            pars.update(model.named_vars)

        elif isinstance(pars, pt.TensorVariable):  # a single variable
            # Note: this has to go before the next clause because
            # TensorVariable instances are iterable...
            pars = {pars.name: pars}

        else:
            try:
                pars = dict(pars)  # try to coerce to a dictionary
            except Exception:
                # if that fails, assume it is an iterable, like a list or tuple
                try:
                    pars = {p.name: p for p in pars}
                except Exception as e:
                    msg = (
                        "Invalid input parameters: The input `pars` must either be a "
                        "dictionary, list, or a single pymc variable, not a "
                        f"'{type(pars)}'."
                    )
                    raise ValueError(msg) from e

        # Set the number of polynomial trend parameters
        self.poly_trend, self._v_trend_names = validate_poly_trend(poly_trend)

        # Calibration offsets of velocity zero-point
        if v0_offsets is None:
            v0_offsets = []

        try:
            v0_offsets = list(v0_offsets)
        except Exception as e:
            msg = (
                "Constant velocity offsets must be an iterable of pymc variables that "
                "define the priors on each offset term."
            )
            raise TypeError(msg) from e

        self.v0_offsets = v0_offsets
        pars.update({p.name: p for p in self.v0_offsets})

        # Store the names of the default parameters, used for validating input:
        # Note: these are *not* the units assumed internally by the code, but
        # are only used to validate that the units for each parameter are
        # equivalent to these
        self._nonlinear_equiv_units = get_nonlinear_equiv_units()
        self._linear_equiv_units = get_linear_equiv_units(
            self.poly_trend, sb2=self._sb2
        )
        self._v0_offsets_equiv_units = get_v0_offsets_equiv_units(self.n_offsets)
        self._all_par_unit_equiv = {
            **self._nonlinear_equiv_units,
            **self._linear_equiv_units,
            **self._v0_offsets_equiv_units,
        }

        # At this point, pars must be a dictionary: validate that all
        # parameters are specified and that they all have units
        for name in self.par_names:
            if name not in pars:
                msg = (
                    f"Missing prior for parameter '{name}': you must specify a prior "
                    "distribution for all parameters."
                )
                raise ValueError(msg)

            if not hasattr(pars[name], xu.UNIT_ATTR_NAME):
                msg = (
                    f"Parameter '{name}' does not have associated units: Use "
                    "thejoker.units to specify units for your pymc variables. See the "
                    "documentation for examples: thejoker.rtfd.io"
                )
                raise ValueError(msg)

            equiv_unit = self._all_par_unit_equiv[name]
            if not getattr(pars[name], xu.UNIT_ATTR_NAME).is_equivalent(equiv_unit):
                msg = (
                    f"Parameter '{name}' has an invalid unit: The units for this "
                    f"parameter must be transformable to '{equiv_unit}'"
                )
                raise ValueError(msg)

        # Enforce that the priors on all linear parameters are Normal (or a
        # subclass of Normal)
        for name in list(self._linear_equiv_units.keys()) + list(
            self._v0_offsets_equiv_units.keys()
        ):
            p = pars[name]
            if not hasattr(p, "owner"):
                msg = f"Invalid type for prior on linear parameter {name}: {type(p)}"
                raise TypeError(msg)

            if not isinstance(
                p.owner.op, pt.random.op.RandomVariable
            ) or p.owner.op._print_name[0] not in ["Normal", "FixedCompanionMass"]:
                msg = (
                    "Priors on the linear parameters (K, v0, etc.) must be independent "
                    f"Normal distributions, not '{p.owner.op._print_name[0]}' (for "
                    f"{name})"
                )
                raise ValueError(msg)

        self.pars = pars

    @classmethod
    def default(
        cls,
        P_min=None,
        P_max=None,
        sigma_K0=None,
        P0=1 * u.year,
        sigma_v=None,
        s=None,
        poly_trend=1,
        v0_offsets=None,
        model=None,
        pars=None,
    ):
        r"""
        An alternative initializer to set up the default prior for The Joker.

        The default prior is:

        .. math::

            p(P) \propto \frac{1}{P} \quad ; \quad P \in (P_{\rm min}, P_{\rm max})\\
            p(e) = B(a_e, b_e)\\
            p(\omega) = \mathcal{U}(0, 2\pi)\\
            p(M_0) = \mathcal{U}(0, 2\pi)\\
            p(s) = 0\\
            p(K) = \mathcal{N}(K \,|\, \mu_K, \sigma_K)\\
            \sigma_K = \sigma_{K, 0} \, \left(\frac{P}{P_0}\right)^{-1/3} \, \left(1 - e^2\right)^{-1/2}

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
        model : `pymc.Model` (optional)
            If not specified, this will create a model instance and store it on
            the prior object.
        pars : dict, list (optional)
            Either a list of pymc variables, or a dictionary of variables with
            keys set to the variable names. If any of these variables are
            defined as deterministic transforms from other variables, see the
            next parameter below.
        """

        model = _validate_model(model)

        nl_pars = default_nonlinear_prior(P_min, P_max, s=s, model=model, pars=pars)
        l_pars = default_linear_prior(
            sigma_K0=sigma_K0,
            P0=P0,
            sigma_v=sigma_v,
            poly_trend=poly_trend,
            model=model,
            pars=pars,
        )

        pars = {**nl_pars, **l_pars}
        return cls(pars=pars, model=model, poly_trend=poly_trend, v0_offsets=v0_offsets)

    @property
    def par_names(self):
        return (
            list(self._nonlinear_equiv_units.keys())
            + list(self._linear_equiv_units.keys())
            + list(self._v0_offsets_equiv_units)
        )

    @property
    def par_units(self):
        return {
            p.name: getattr(p, xu.UNIT_ATTR_NAME, u.one) for _, p in self.pars.items()
        }

    @property
    def n_offsets(self):
        return len(self.v0_offsets)

    def __repr__(self):
        return f'<JokerPrior [{", ".join(self.par_names)}]>'

    def __str__(self):
        return ", ".join(self.par_names)

    def _get_raw_samples(
        self,
        size=1,
        generate_linear=False,
        return_logprobs=False,
        rng=None,
        dtype=None,
        **kwargs,
    ):
        if dtype is None:
            dtype = np.float64

        sub_pars = {
            k: p
            for k, p in self.pars.items()
            if k in self._nonlinear_equiv_units
            or (
                (k in self._linear_equiv_units or k in self._v0_offsets_equiv_units)
                and generate_linear
            )
        }

        # MAJOR HACK RELATED TO UPSTREAM ISSUES WITH pymc:
        # init_shapes = {}
        # for name, par in sub_pars.items():
        #     if hasattr(par, "distribution"):
        #         init_shapes[name] = par.distribution.shape
        #         par.distribution.shape = (size,)

        par_names = list(sub_pars.keys())
        par_list = [sub_pars[k] for k in par_names]
        samples_values = pm.draw(par_list, draws=size, random_seed=rng)

        raw_samples = {
            name: samples.astype(dtype)
            for name, p, samples in zip(par_names, par_list, samples_values)
        }

        if return_logprobs:
            # raise NotImplementedError("This feature has been disabled in v1.3")
            logp = []
            for par in sub_pars.values():
                try:
                    _logp = pm.logp(par, raw_samples[par.name]).eval()
                except Exception:
                    logger.warning(
                        f"Cannot auto-compute log-prior value for parameter {par}"
                    )
                    continue

                logp.append(_logp)
            log_prior = np.sum(logp, axis=0)
        else:
            log_prior = None

        # CONTINUED MAJOR HACK RELATED TO UPSTREAM ISSUES WITH pymc:
        # for name, par in sub_pars.items():
        #     if hasattr(par, "distribution"):
        #         par.distribution.shape = init_shapes[name]

        return raw_samples, sub_pars, log_prior

    @deprecated_renamed_argument(
        "random_state", "rng", since="v1.3", warning_type=DeprecationWarning
    )
    def sample(
        self,
        size=1,
        generate_linear=False,
        return_logprobs=False,
        rng=None,
        dtype=None,
        **kwargs,
    ):
        """
        Generate random samples from the prior.

        .. note::

            Right now, generating samples with the prior values is slow (i.e.
            with ``return_logprobs=True``) because of pymc3 issues (see
            discussion here:
            https://discourse.pymc.io/t/draw-values-speed-scaling-with-transformed-variables/4076).
            This will hopefully be resolved in the future...

        Parameters
        ----------
        size : int (optional)
            The number of samples to generate.
        generate_linear : bool (optional)
            Also generate samples in the linear parameters.
        return_logprobs : bool (optional)
            Generate the log-prior probability at the position of each sample.
        **kwargs
            Additional keyword arguments are passed to the
            `~thejoker.JokerSamples` initializer.

        Returns
        -------
        samples : `thejoker.Jokersamples`
            The random samples.

        """
        from thejoker.samples import JokerSamples

        raw_samples, sub_pars, log_prior = self._get_raw_samples(
            size, generate_linear, return_logprobs, rng, dtype, **kwargs
        )

        if generate_linear:
            par_names = self.par_names
        else:
            par_names = list(self._nonlinear_equiv_units.keys())

        # Apply units if they are specified:
        prior_samples = JokerSamples(
            poly_trend=self.poly_trend, n_offsets=self.n_offsets, **kwargs
        )
        for name in par_names:
            p = sub_pars[name]
            unit = getattr(p, xu.UNIT_ATTR_NAME, u.one)

            if name not in prior_samples._valid_units:
                continue

            prior_samples[name] = np.atleast_1d(raw_samples[name]) * unit

        if return_logprobs:
            prior_samples["ln_prior"] = log_prior

        # TODO: right now, elsewhere, we assume the log_prior is a single value
        # for each sample (i.e. the total prior value). In principle, we could
        # store all of the individual log-prior values (for each parameter),
        # like here:
        # log_prior = {k: np.atleast_1d(v)
        #              for k, v in log_prior.items()}
        # log_prior = Table(log_prior)[par_names]

        return prior_samples


@u.quantity_input(P_min=u.day, P_max=u.day)
def default_nonlinear_prior(P_min=None, P_max=None, s=None, model=None, pars=None):
    r"""
    Retrieve pymc variables that specify the default prior on the nonlinear
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
    P_min : `astropy.units.Quantity` [time]
    P_max : `astropy.units.Quantity` [time]
    s : `TensorVariable`, `astropy.units.Quantity` [speed]
    model : `pymc.Model`
        This is either required, or this function must be called within a pymc
        model context.
    """
    from pymc_ext.distributions import angle

    from .distributions import Kipping13Global, UniformLog

    model = pm.modelcontext(model)

    if pars is None:
        pars = {}

    if s is None:
        s = 0 * u.m / u.s

    if isinstance(s, pt.TensorVariable):
        pars["s"] = pars.get("s", s)
    else:
        if not hasattr(s, "unit") or not s.unit.is_equivalent(u.km / u.s):
            raise u.UnitsError("Invalid unit for s: must be equivalent to km/s")

    # dictionary of parameters to return
    out_pars = {}

    with model:
        # Set up the default priors for parameters with defaults

        # Note: we have to do it this way (as opposed to with .get(..., default)
        # because this can only get executed if the param is not already
        # defined, otherwise variables are defined twice in the model
        if "e" not in pars:
            out_pars["e"] = xu.with_unit(Kipping13Global("e"), u.one)

        # If either omega or M0 is specified by user, default to U(0,2π)
        if "omega" not in pars:
            out_pars["omega"] = xu.with_unit(angle("omega"), u.rad)

        if "M0" not in pars:
            out_pars["M0"] = xu.with_unit(angle("M0"), u.rad)

        if "s" not in pars:
            out_pars["s"] = xu.with_unit(
                pm.Deterministic("s", pt.constant(s.value)), s.unit
            )

        if "P" not in pars:
            if P_min is None or P_max is None:
                msg = (
                    "If you are using the default period prior, you must pass in both "
                    "P_min and P_max to set the period prior domain."
                )
                raise ValueError(msg)
            out_pars["P"] = xu.with_unit(
                UniformLog("P", P_min.value, P_max.to_value(P_min.unit)), P_min.unit
            )

    for k in pars:
        out_pars[k] = pars[k]

    return out_pars


@u.quantity_input(sigma_K0=u.km / u.s, P0=u.day)
def default_linear_prior(
    sigma_K0=None, P0=None, sigma_v=None, poly_trend=1, model=None, pars=None
):
    r"""
    Retrieve pymc variables that specify the default prior on the linear
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
    model : `pymc.Model`
        This is either required, or this function must be called within a pymc
        model context.
    """
    from .distributions import FixedCompanionMass

    model = pm.modelcontext(model)

    if pars is None:
        pars = {}

    # dictionary of parameters to return
    out_pars = {}

    # set up poly. trend names:
    poly_trend, v_names = validate_poly_trend(poly_trend)

    # get period/ecc from dict of nonlinear parameters
    P = model.named_vars.get("P", None)
    e = model.named_vars.get("e", None)
    if P is None or e is None:
        raise ValueError(
            "Period P and eccentricity e must both be defined as "
            "nonlinear parameters on the model."
        )

    if v_names and "v0" not in pars:
        sigma_v = validate_sigma_v(sigma_v, poly_trend, v_names)

    with model:
        if "K" not in pars:
            if sigma_K0 is None or P0 is None:
                raise ValueError(
                    "If using the default prior form on K, you "
                    "must pass in a variance scale (sigma_K0) "
                    "and a reference period (P0)"
                )

            # Default prior on semi-amplitude: scales with period and
            # eccentricity such that it is flat with companion mass
            v_unit = sigma_K0.unit
            out_pars["K"] = xu.with_unit(
                FixedCompanionMass("K", P=P, e=e, sigma_K0=sigma_K0, P0=P0), v_unit
            )
        else:
            v_unit = getattr(pars["K"], xu.UNIT_ATTR_NAME, u.one)

        for i, name in enumerate(v_names):
            if name not in pars:
                # Default priors are independent gaussians
                # FIXME: make mean, mu_v, customizable
                out_pars[name] = xu.with_unit(
                    pm.Normal(name, 0.0, sigma_v[name].value), sigma_v[name].unit
                )

    for k in pars.keys():
        out_pars[k] = pars[k]

    return out_pars
