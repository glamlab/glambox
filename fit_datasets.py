#!usr/bin/python
import numpy as np
import pandas as pd
import pymc3 as pm

import glambox as gb

data = pd.read_csv('resources/data_aggregate.csv')
data['rt'] = data['rt'] / 1000

estimates = []

for dataset in data['dataset'].unique():
    # subset by dataset
    data_d = data.loc[data['dataset'] == dataset].copy()
    # remove empty columns (for 2-item datasets)
    data_d = data_d.dropna(axis=1)

    for subject in data_d['subject'].unique():
        data_ds = data_d.loc[data_d['subject'] == subject].copy()
        data_ds['subject'] = 0
        glam = gb.GLAM(data=data_ds)
        glam.make_model(kind='individual')
        glam.fit(method='MCMC', tune=2000, draws=2000)
        summary = pm.summary(glam.trace[0])
        estimates_s = pd.DataFrame(dict(
            subject=subject,
            dataset=dataset,
            v=summary.loc['v__0_0', 'mean'],
            gamma=summary.loc['gamma__0_0', 'mean'],
            s=summary.loc['s__0_0', 'mean'],
            tau=summary.loc['tau__0_0', 'mean']),
                                   index=np.ones(1) * subject)
        estimates.append(estimates_s)
        pd.concat(estimates).to_csv('individual_estimates_sec_nhb2019.csv', index=False)
estimates = pd.concat(estimates)
estimates.to_csv('individual_estimates_sec_nhb2019.csv', index=False)