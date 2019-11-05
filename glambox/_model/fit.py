#!/usr/bin/python

import pymc3 as pm
import numpy as np


def fit_models(models,
               method='MCMC',
               verbose=True,
               draws=2000,
               n_vi=200000,
               step=pm.Metropolis,
               **kwargs):
    """
    Fit GLAM model instance(s)

    Input
    ---
    models : PyMC3 model instance(s)
        single GLAM model instance
        or list of multiple models

    method : string, optional
        whether to fit the model(s) by means
        of MCMC sampling ('MCMC') or
        variational inference ('VI')

    verbose : bool, optional
        whether to print updates on
        fitting progress

    draws : int, optional
        number of posterior samples to draw; 
        only, if method = 'MCMC'

    n_vi : int, optional
        number of VI samples
        only, if method = 'VI'

    step : PyMC3 step method, optional
        step method to use for MCMC sampling
        only, if method = 'MCMC'
        e.g. [pm.Metropolis, pm.NUTS]
        see PyMC3 documentation for a complete
        overview of available step methods

    Returns
    ---
    traces : PyMC3 trace(s)
        list of resulting PyMC3 model traces

    """
    if isinstance(models, pm.model.Model):
        models = [models]
    elif isinstance(models, list) and np.alltrue(
            np.array([isinstance(model, pm.model.Model) for model in models])):
        pass
    else:
        raise ValueError(
            "Models must be list of <pymc3.model.Model> instances.")
    if verbose:
        print('Fitting {} model(s) using {}...'.format(len(models), method))
    traces = []

    for m, model in enumerate(models):
        if verbose:
            print('  Fitting model {} of {}...'.format(m + 1, len(models)))
        with model:
            if method == 'MCMC':
                if step is not None:
                    step_method = step()
                else:
                    step_method = None
                trace = pm.sample(draws=draws, step=step_method, **kwargs)
            elif method == 'VI':
                vi_est = pm.fit(n=n_vi, **kwargs)
                trace = vi_est.sample(draws)
            else:
                raise ValueError("Method must be 'MCMC' or 'VI'.")
            traces.append(trace)

    return traces
