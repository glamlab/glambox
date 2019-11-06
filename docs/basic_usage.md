# Basic Usage

## Data format, the `GLAM` class

The core functionality of the GLAMbox is implemented in the `GLAM` model class. To apply the GLAM to data, an instance of the model class needs to be instantiated and supplied with the experimental data, first:

```python
import glambox as gb
glam = gb.GLAM(data=data)
```

The data must be a pandas (McKinney, 2010) DataFrame with one row per trial, containing the following variable entries:

- `subject`: Subject index (`int`, starting with 0)
- `trial`: Trial index (`int`, starting with 0)
- `choice`: Chosen item (`int`, items should be 0, 1, ..., N)
- `rt`: Response time (`float`, in seconds)
- for each item *i* in the choice set:
    - `item_value_i`: The item value (`float`)
    - `gaze_i`: The fraction of total time in this trial that the participant spent looking at this item (`float`, between 0 and 1)
- additional variables coding groups or conditions (`str` or `int`)

For reference, the first two rows of a pandas DataFrame ready to be used with GLAMbox could look like this:

| subject | trial | choice | rt   | item_value_0 | item_value_1 | item_value_2 | gaze_0 | gaze_1 | gaze_2 | speed  |
|---------|-------|--------|------|--------------|--------------|--------------|--------|--------|--------|--------|
| 0       | 0     | 0      | 2.41 | 6            | 4            | 3            | 0.56   | 0.22   | 0.22   | 'fast' |
| 0       | 1     | 1      | 3.65 | 5            | 5            | 3            | 0.25   | 0.34   | 0.41   | 'slow' |

Next, the respective PyMC3 model, which will later be used to estimate the model's parameters, can be built using the `make_model` method. Here, the researcher specifies the kind of the model: `'individual'` if the parameters should be estimated for each subject individually, `'hierarchical'` for hierarchical parameter estimation, or `'pooled'` to estimate a single parameter set for all subjects. At this stage, the researcher can also specify experimental parameter dependencies: For example, a parameter could be expected to vary between groups or conditions. In line with existing modeling toolboxes (e.g., Voss & Voss, 2007; Wiecki, Sofer, Frank, 2013) dependencies are defined using the `depends_on` argument. `depends_on` expects a dictionary with parameters as keys and experimental factors as values (e.g., `depends_on=dict(v='speed')` for factor `'speed'` with conditions `'fast'` and `'slow'` in the data). The toolbox internally handles within- and between subject designs and assigns parameters accordingly. If multiple conditions are given for a factor, one parameter will be designated for each condition. Finally, the `make_model` method allows parameters to be fixed to a specific value using the `*_val` arguments (e.g., `gamma_val=1` for a model without gaze bias). If parameters should be fixed for individual subjects, a list of individual values needs to be passed.

```python
model.make_model(kind='individual',
                 depends_on=dict(v='speed'),
                 gamma_val=1)
```

## Inference

Once the PyMC3 model is built, parameters can be estimated using the `fit` method:

```python
model.fit(method='MCMC',
          tune=5000,
          draws=5000)
```

The `fit` method defaults to Metropolis Hastings Markov-Chain-Monte-Carlo (MCMC) sampling, but also allows for Variational Inference.

## Accessing parameter estimates

After parameter estimation is completed, the resulting estimates can be accessed with the `estimates` attribute of the GLAM model instance. This returns a table with one row for each set of parameter estimates for each individual and condition in the data. For each parameter, a maximum a posteriori (MAP) estimate is given, in addition to the 95\% Highest-Posterior Density Interval (HPD). If the parameters were estimated hierarchically, the table also contains estimates of the group-level parameters. 

## Predicting choices and response times

Choices and RTs can be predicted with the GLAM by the use of the `predict`method:

```python
model.predict(n_repeats=50)
```

For each trial of the dataset that is attached to the model instance, this method predicts a choice and RT using the previously determined MAP parameter estimates. To obtain a stable estimate of the GLAM's predictions, as well as the noise contained within them, it is recommended to repeat every trial multiple times during the prediction. The number of trial repeats can be specified with the `n_repeats` argument. After the prediction is completed, the predicted data can be accessed with the `prediction` attribute of the model.

## References
- McKinney, W. (2010, June). Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51-56).
- Thomas, A. W., Molter, F., Krajbich, I., Heekeren, H. R., & Mohr, P. N. (2019). Gaze bias differences capture individual choice behaviour. Nature human behaviour, 3(6), 625.
- Voss, A., & Voss, J. (2007). Fast-dm: A free program for efficient diffusion model analysis. Behavior Research Methods, 39(4), 767-775.
- Wiecki, T. V., Sofer, I., & Frank, M. J. (2013). HDDM: Hierarchical Bayesian estimation of the drift-diffusion model in Python. Frontiers in neuroinformatics, 7, 14.
