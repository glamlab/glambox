# GLAMbox

GLAMbox is a Python toolbox for investigating the association between gaze allocation and decision behaviour, and applying the Gaze-weighted Linear Accumulator Model ([Thomas, Molter et al., 2019](https://www.nature.com/articles/s41562-019-0584-8)).

See the [BioRxiv preprint](https://www.biorxiv.org/content/10.1101/741678v1) for detailed background, model description and example applications.

A full documentation page of the toolbox is available at [https://glambox.readthedocs.io](https://glambox.readthedocs.io).

## Installation

GLAMbox is written for Python 3.7 and requires a working Python environment running on your computer. We recommend to install the [Anaconda Distribution](https://www.anaconda.com/distribution/) (available for all major platforms).
With the Python environment fully set up, the GLAMbox module can be installed from the command line using pip:

```bash
pip install glambox
```

This command also installs all of GLAMbox's dependencies, which are listed in the `requirements.txt`.

## Quickstart

Fitting the GLAM to a dataset can be done in just a few lines of code:

```py
import glambox as gb
import pandas as pd

# load dataset (format must be GLAMbox compatible, of course)
data = pd.read_csv('data.csv')

# create the GLAM model object
model = gb.GLAM(data)

# build the PyMC3 model
model.make_model(kind='individual')

# perform MCMC sampling
model.fit()

# inspect parameter estimates
print(model.estimates)

# predict data using MAP estimates, save predictions
model.predict()
model.prediction.to_csv('prediction.csv')
```

A more detailed overview of the available functions can be found in the [Basic usage section](#basic-usage) and the [API Reference](https://glambox.readthedocs.io/en/latest/glambox.html).

## Application Examples

This repository includes Jupyter notebooks with full code for three usage examples outlined in the [BioRxiv preprint](https://www.biorxiv.org/content/10.1101/741678v1). The notebook files for these examples can be found in the `examples` folder of this repository. They can be downloaded and run interactively with [Jupyter](https://www.jupyter.org) (also included in the Anaconda Python distribution). Fully rendered (non-interactive) html versions can be found on the [documentation page](https://glambox.readthedocs.io).

### Example 1: Individual gaze biases

In this example, we demonstrate individual model fitting, model comparisons between model variants, and out-of-sample prediction. ([Jupyter](https://github.com/glamlab/glambox/blob/master/Example_1_Individual_estimation.ipynb), [html](https://glambox.readthedocs.io/en/latest/examples/Example_1_Individual_estimation.html))

### Example 2: Hierarchical parameter estimation

In the second example, we demonstrate how to setup a hierarchical model with multiple groups, and compare parameter estimates between groups. ([Jupyter](https://github.com/glamlab/glambox/blob/master/Example_2_Hierarchical_estimation.ipynb), [html](https://glambox.readthedocs.io/en/latest/examples/Example_2_Hierarchical_estimation.html))

### Example 3: Parameter Recovery

In the last example, we demonstrate how to perform a basic parameter recovery analyis for a given dataset, using GLAMbox. ([Jupyter](https://github.com/glamlab/glambox/blob/master/Example_3_Parameter_recovery.ipynb), [html](https://glambox.readthedocs.io/en/latest/examples/Example_3_Parameter_recovery.html))

## Basic usage

### Data format, the `GLAM` class

The core functionality of the GLAMbox is implemented in the `GLAM` model class. To apply the GLAM to data, an instance of the model class needs to be instantiated and supplied with the experimental data, first:

```python
import glambox as gb
glam = gb.GLAM(data=data)
````

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

### Inference

Once the PyMC3 model is built, parameters can be estimated using the `fit` method:

```python
model.fit(method='MCMC',
          tune=5000,
          draws=5000)
```

The `fit` method defaults to Metropolis Hastings Markov-Chain-Monte-Carlo (MCMC) sampling, but also allows for Variational Inference.

### Accessing parameter estimates

After parameter estimation is completed, the resulting estimates can be accessed with the `estimates` attribute of the GLAM model instance. This returns a table with one row for each set of parameter estimates for each individual and condition in the data. For each parameter, a maximum a posteriori (MAP) estimate is given, in addition to the 95\% Highest-Posterior Density Interval (HPD). If the parameters were estimated hierarchically, the table also contains estimates of the group-level parameters. 



### Comparing parameters between groups or conditions

Parameter estimates can be compared between different experimental groups or conditions (specified with the `depends_on` keyword when calling `make_model`) using the `compare_parameters` function from the `analysis` module. It takes as input the fitted GLAM instance, a list of parameters (`'v'`, `'s'`, `'gamma'`, `'tau'`), and a list of pairwise comparisons between groups or conditions. The comparison argument expects a list of tuples (e.g., `[('group1', 'group2'), ('group1', 'group3')]`). For example, given a fitted model instance (here `glam`) a comparison of the $\gamma$ parameter between two groups (`group1` and `group2`) can be computed as: 

```py
from gb.analysis import compare_parameters
comparison = compare_parameters(model=glam, 
                                parameters=['gamma'],
                                comparisons=[('group1', 'group2')])
```

The function then returns a table with one row per specified comparison, and columns containing the mean posterior difference, percentage of the posterior above zero, and corresponding 95\% HPD interval. If supplied with a hierarchical model, the function computes differences between group-level parameters. If an individual type model is given, it returns comparison statistics for each individual.

Comparisons can  be visualized using the `compare_parameters` function from the `plots` module. It takes the same input as its analogue in the `analysis` module. It plots posterior distributions of parameters and the posterior distributions of any differences specified using the `comparisons` argument. For a usage example and plot see [usage example 2](https://glambox.readthedocs.io/en/latest/examples/Example_2_Hierarchical_estimation.html) in the full documentation.

### Comparing model variants

Model comparisons between multiple GLAM variants (e.g., full and restricted variants) can be performed using the `compare_models` function, which wraps the function of the same name from the PyMC3 library. The `compare_models` function takes as input a list of fitted model instances that are to be compared. Additional keyword arguments can be given and are passed on to the underlying PyMC3 `compare` function. This allows the user, for example, to specify the information criterion used for the comparison via the `ic` argument (`'WAIC'` or `'LOO'` for Leave-One-Out cross validation). It returns a table containing an estimate of the specified information criterion, standard errors, difference to the best-fitting model, standard error of the difference, and other output variables from PyMC3 for each inputted model (and subject, if individually estimated models were given). We refer the reader to [usage example 1](https://glambox.readthedocs.io/en/latest/examples/Example_1_Individual_estimation.html) in the full documentation for the full code and exemplary output from the `compare_models` function.

### Predicting choices and response times

Choices and RTs can be predicted with the GLAM by the use of the `predict`method:

```python
model.predict(n_repeats=50)
```

For each trial of the dataset that is attached to the model instance, this method predicts a choice and RT using the previously determined MAP parameter estimates. To obtain a stable estimate of the GLAM's predictions, as well as the noise contained within them, it is recommended to repeat every trial multiple times during the prediction. The number of trial repeats can be specified with the `n_repeats` argument. After the prediction is completed, the predicted data can be accessed with the `prediction` attribute of the model.

# References
- McKinney, W. (2010, June). Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51-56).
- Thomas, A. W., Molter, F., Krajbich, I., Heekeren, H. R., & Mohr, P. N. (2019). Gaze bias differences capture individual choice behaviour. Nature human behaviour, 3(6), 625.
- Voss, A., & Voss, J. (2007). Fast-dm: A free program for efficient diffusion model analysis. Behavior Research Methods, 39(4), 767-775.
- Wiecki, T. V., Sofer, I., & Frank, M. J. (2013). HDDM: Hierarchical Bayesian estimation of the drift-diffusion model in Python. Frontiers in neuroinformatics, 7, 14.
