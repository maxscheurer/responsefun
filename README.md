ResponseFun
==============================

Fun with Response Functions in the Algebraic Diagrammatic Construction Framework.


### Installation

You can install `responsefun` with Conda using the conda-forge channel:

```bash
conda install responsefun -c conda-forge
```


### Development

A suitable `conda` environment for development can be created from the `ci_env.yml` file.
After cloning the repository and navigating to the responsefun directory, the `responsefun` python package can be installed with the following command:
`pip install -e .`

Tests can be run with `pytest responsefun`, or `pytest --pyargs responsefun` if the package is installed.
To exclude the slowest test, run `pytest -m "not slow" responsefun`.

Code style is enforced through `black` (formatting), `isort` (sorting import statements), and `ruff` (linting).

### Citation

If you use `responsefun`, please cite [our article in JCTC.](https://doi.org/10.1021/acs.jctc.3c00456)
### Copyright

Copyright (c) 2023, The `responsefun` Developers


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
