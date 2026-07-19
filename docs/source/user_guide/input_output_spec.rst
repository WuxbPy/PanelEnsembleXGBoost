.. _input_output_spec:

Input/Output Specifications
===========================

.. warning::
   **Critical:** The model expects **raw panel data**—i.e., the feature
   DataFrame ``X`` should NOT already include random effects grouping columns.
   These must be passed separately via the ``groups_for_random_effects``
   parameter. Duplicate columns across ``X`` and ``groups_for_random_effects``
   will trigger a warning and the columns from ``X`` will be used.

Fit Method
----------

:meth:`PanelEnsembleXGBoost.fit`

.. list-table:: **Input: X**
   :header-rows: 1
   :widths: 20 80

   * - Parameter Type
     - Details
   * - Type
     - ``pd.DataFrame``
   * - Shape
     - ``(n_samples, n_features)``
   * - Content
     - Numeric feature columns only. Categorical features should be
       one-hot encoded or passed as grouping variables.
   * - Notes
     - - Must NOT include random effects grouping columns
       - Must NOT include time/individual identifier columns that would
         leak panel structure (these go into ``groups_for_random_effects``)
       - Missing values should be handled before fitting

.. list-table:: **Input: y**
   :header-rows: 1
   :widths: 20 80

   * - Parameter Type
     - Details
   * - Type
     - ``pd.Series`` or ``np.ndarray``
   * - Shape
     - ``(n_samples,)``
   * - Format by family
     - **Gaussian**: Continuous real values
     - **Binomial**: Binary labels (0/1 or string classes; auto-encoded)
     - **Multinomial**: Integer or string class labels (auto-encoded)

.. list-table:: **Input: groups_for_random_effects**
   :header-rows: 1
   :widths: 20 80

   * - Parameter Type
     - Details
   * - Type
     - ``pd.DataFrame`` or ``np.ndarray``
   * - Shape
     - ``(n_samples, n_groups)`` or ``(n_samples,)``
   * - Content
     - Grouping variable column(s), e.g., ``{'region': [...]}`` or
       ``{'firm': [...], 'year': [...]}``
   * - Notes
     - If ``X`` already contains these columns, they will be used from ``X``
       and a warning is emitted.

.. list-table:: **Output**
   :header-rows: 1
   :widths: 20 80

   * - Parameter Type
     - Details
   * - Type
     - ``PanelEnsembleXGBoost`` (returns ``self`` for chaining)
   * - Side effects
     - Stores fitted sub-models in instance attributes
     - Sets ``self.is_fitted = True``

Predict Method
--------------

:meth:`PanelEnsembleXGBoost.predict`

.. list-table:: **Input: X**
   :header-rows: 1
   :widths: 20 80

   * - Parameter Type
     - Details
   * - Type
     - ``pd.DataFrame``
   * - Shape
     - ``(n_samples, n_features)``
   * - Notes
     - Must have the same feature columns (and order) as training
     - Grouping columns are not required in ``X`` if provided via
       ``groups_for_random_effects``

.. list-table:: **Input: groups_for_random_effects** (optional)
   :header-rows: 1
   :widths: 20 80

   * - Parameter Type
     - Details
   * - Type
     - ``pd.DataFrame``
   * - Shape
     - ``(n_samples, n_groups)``
   * - Notes
     - If training data had groups, prediction data must provide the same
       group identifiers to compute random effects

.. list-table:: **Output**
   :header-rows: 1
   :widths: 20 80

   * - Parameter Type
     - Details
   * - Type
     - ``np.ndarray``
   * - Shape
     - ``(n_samples,)``
   * - Format by family
     - **Gaussian**: Continuous values (original scale)
     - **Binomial**: Binary labels (0/1)
     - **Multinomial**: Integer class labels

predict_proba Method
--------------------

:meth:`PanelEnsembleXGBoost.predict_proba`

.. list-table:: **Input**
   :header-rows: 1
   :widths: 20 80

   * - Parameter Type
     - Details
   * - Signature
     - Same as ``predict()``
   * - Available for
     - Classification families only (``binomial``, ``multinomial``)
   * - Raises
     - ``AttributeError`` if called for a regression family

.. list-table:: **Output**
   :header-rows: 1
   :widths: 20 80

   * - Parameter Type
     - Details
   * - Type
     - ``np.ndarray``
   * - Shape (binomial)
     - ``(n_samples,)`` — probability of positive class
   * - Shape (multinomial)
     - ``(n_samples, n_classes)`` — probability for each class

get_stage_predictions
---------------------

:meth:`PanelEnsembleXGBoost.get_stage_predictions`

.. list-table:: **Input**
   :header-rows: 1
   :widths: 20 80

   * - Same as ``predict()``
     - 

.. list-table:: **Output**
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``'stage1_random_effects'``
     - :math:`\hat{y}_1` — random/mixed effects predictions (link space)
   * - ``'stage2_smooth_effects'``
     - :math:`\hat{y}_2` — GAM/B-spline smooth effects (link space)
   * - ``'stage3_nonlinear_effects'``
     - :math:`\hat{y}_3` — XGBoost nonlinear interaction effects (link space)
   * - ``'total'``
     - :math:`\hat{y}_1 + \hat{y}_2 + \hat{y}_3` (link space)

Common Pitfalls
---------------

.. warning:: **Duplicate columns**

   If a grouping column (e.g., ``'region'``) is present in both ``X`` and
   ``groups_for_random_effects``, the model will issue a warning and use
   the version from ``X``. This can lead to subtle bugs. Best practice:
   keep the feature DataFrame free of grouping columns.

.. warning:: **Missing groups at prediction time**

   If a new group (not seen during training) appears in test data,
   its random effect is set to zero. The model will still produce
   predictions using only fixed effects.

.. warning:: **Small number of groups**

   - < 3 groups: Random effects stage is skipped entirely
   - < 5 groups: A caution is raised; variance estimates may be unreliable
