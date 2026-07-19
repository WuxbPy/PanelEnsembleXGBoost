Contributing
============

Contributions are welcome! Please follow these guidelines:

1. **Bug reports**: Open an issue on GitHub with a minimal reproduction script
2. **Feature requests**: Describe the use case and proposed API
3. **Pull requests**: Ensure all existing tests pass, add tests for new features,
   and update documentation

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/WuxbPy/PanelEnsembleXGBoost.git
   cd PanelEnsembleXGBoost
   pip install -r requirements.txt
   pip install -r docs/requirements.txt  # For docs

Building Documentation Locally
------------------------------

.. code-block:: bash

   cd docs
   sphinx-build -b html source build
   # Open build/index.html in your browser
