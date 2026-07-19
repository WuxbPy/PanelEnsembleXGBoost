.. _distribution_families:

Distribution Families
=====================

PanelEnsembleXGBoost supports multiple distribution families through a
**strategy pattern**, allowing each family to define its own:

* Link function and inverse link function
* Residual computation
* Deviance (goodness-of-fit) measure
* Stage-appropriate model classes (GAM, XGBoost objective)

Supported Families
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 40

   * - Family
     - Task Type
     - Link Function
     - Example Use Case
   * - ``'gaussian'``
     - Regression (continuous)
     - Identity (:math:`g(\mu) = \mu`)
     - Investment, wages, GDP, prices
   * - ``'binomial'``
     - Binary classification
     - Logit (:math:`g(\mu) = \log\frac{\mu}{1-\mu}`)
     - Default/Churn, disease diagnosis
   * - ``'multinomial'``
     - Multiclass classification
     - Softmax
     - Industry classification, voting outcomes

Usage
-----

.. code-block:: python

   # Gaussian regression (default)
   model = PanelEnsembleXGBoost(family='gaussian')
   
   # Binary classification
   model = PanelEnsembleXGBoost(family='binomial')

   # Multiclass classification
   model = PanelEnsembleXGBoost(family='multinomial')

Stage Adaptation by Family
--------------------------

Each family automatically selects the appropriate model at each stage:

.. list-table::
   :header-rows: 1
   :widths: 15 25 25 30

   * - Family
     - Stage 1 (Random Effects)
     - Stage 2 (Smooth)
     - Stage 3 (XGBoost)
   * - ``gaussian``
     - LMM (custom PQL)
     - LinearGAM / LinearRegression
     - XGBRegressor (``reg:squarederror``)
   * - ``binomial``
     - GLMM (PQL, logit link)
     - LogisticGAM / LogisticRegression
     - XGBRegressor (link space, ``reg:squarederror``)
   * - ``multinomial``
     - GLMM (PQL, softmax link)
     - LogisticGAM / LogisticRegression
     - XGBClassifier (``multi:softprob``)

Link Function Space
-------------------

For non-Gaussian families, all three stages operate in the **link function
space** (also called linear predictor space). This is a critical design
decision:

- **Stage 1** outputs are in link space (log-odds for binomial)
- **Stage 2** smooth terms are fitted in link space
- **Stage 3** XGBoost predicts in link space
- **Final sum** is transformed through the inverse link function to produce
  predictions in the response space

This ensures that the additive decomposition :math:`\eta = \hat{y}_1 + \hat{y}_2 + \hat{y}_3`
is mathematically valid and interpretable.

Family Strategy API
-------------------

Each family is implemented as a subclass of :class:`family_strategies.FamilyStrategy`:

.. autoclass:: family_strategies.FamilyStrategy
   :members:
   :noindex:

Concrete implementations:

.. autoclass:: family_strategies.GaussianStrategy
   :members:
   :noindex:

.. autoclass:: family_strategies.BinomialStrategy
   :members:
   :noindex:

.. autoclass:: family_strategies.MultinomialStrategy
   :members:
   :noindex:

.. autoclass:: family_strategies.PoissonStrategy
   :members:
   :noindex:

.. autoclass:: family_strategies.GammaStrategy
   :members:
   :noindex:
