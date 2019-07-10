#!/usr/bin/python

import pymc3 as pm
import numpy as np


def fit_models(models, method='MCMC', verbose=True, draws=2000, n_vi=200000, step=pm.Metropolis, **kwargs):
    if isinstance(models, pm.model.Model):
        models = [models]
    elif isinstance(models, list) and np.alltrue(np.array([isinstance(model, pm.model.Model)
                                                           for model in models])):
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
