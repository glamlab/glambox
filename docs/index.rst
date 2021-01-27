GLAMbox
=======

GLAMbox is a Python toolbox for investigating the association between gaze allocation and decision behaviour, and applying the Gaze-weighted Linear Accumulator Model (`Thomas, Molter et al., 2019 <https://www.nature.com/articles/s41562-019-0584-8>`_, `full text available online <https://www.nature.com/articles/s41562-019-0584-8.epdf?author_access_token=xeQN3rCxVzEjDW1KIzHcQtRgN0jAjWel9jnR3ZoTv0Pkem38lvjffFaw7PdV4X2RsPW5yjioXGDc2z-KxEoWWx_Onqu7nmdR2xmxV4sWJOfWwlsBoqzj9rGEA70bYbJZyqrAIz51der3mHG62s1Ltg%3D%3D>`_).

See the `GLAMbox paper <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0226428>`_ for detailed background, model description and example applications.

Installation
^^^^^^^^^^^^

GLAMbox is written for Python 3.7 and requires a working Python environment running on your computer. We recommend to install the `Anaconda Distribution <https://www.anaconda.com/distribution/>`_ (available for all major platforms).
With the Python environment fully set up, the GLAMbox module can be installed from the command line using pip:

.. code-block:: bash

    pip install glambox

This command also installs all of GLAMbox's dependencies, which are listed in the :code:`requirements.txt` file in the `Github repository <https://www.github.com/glamlab/glambox>`_.

.. tip::

    Using ``conda`` you can install ``glambox`` into a new environment, separate from your base environment:
    
    1. Create a new environment named glambox: ``conda create -n glambox python=3.7``
    2. Activate it: ``conda activate glambox``
    3. Install the ``glambox`` module and its dependencies: ``pip install glambox``

Quickstart
^^^^^^^^^^

Fitting the GLAM to a dataset can be done in just a few lines of code:

.. code-block:: python

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


A more detailed overview of the available functions can be found in the :doc:`Basic Usage section <basic_usage>` and the :doc:`API Reference <glambox>`.

Application Examples
^^^^^^^^^^^^^^^^^^^^

This documentation includes the three usage examples outlined in the `GLAMbox paper <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0226428>`_ with full code. The original Jupyter notebook files for these examples can be found in the :code:`examples` folder in the `Github repository <https://www.github.com/glamlab/glambox>`_. When downloaded and run with `Jupyter <https://www.jupyter.org>`_ (also included in the Anaconda Python distribution), the notebooks can be ran interactively.

:doc:`Example 1: Individual gaze biases <examples/Example_1_Individual_estimation>`
------------------------------------------------------------------------------------
In this example, we demonstrate individual model fitting, model comparisons between model variants, and out-of-sample prediction.

:doc:`Example 2: Hierarchical parameter estimation <examples/Example_2_Hierarchical_estimation>`
-------------------------------------------------------------------------------------------------
In the second example, we demonstrate how to setup a hierarchical model with multiple groups, and compare parameter estimates between groups.


:doc:`Example 3: Parameter Recovery <examples/Example_3_Parameter_recovery>`
----------------------------------------------------------------------------
In the last example, we demonstrate how to perform a basic parameter recovery analyis for a given dataset, using GLAMbox.



.. toctree::
   :maxdepth: 2
   :caption: Contents

   Home <self>
   basic_usage
   API Reference <glambox>
   license


.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/Example_1_Individual_estimation
   examples/Example_2_Hierarchical_estimation
   examples/Example_3_Parameter_recovery

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
