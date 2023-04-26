ResponseFun
==============================

Fun with Response Functions in the Algebraic Diagrammatic Construction Framework.

### Development

A suitable `conda` environment for development can be created from the `ci_env.yml` file.

Tests can be run with `pytest responsefun`, or `pytest --pyargs responsefun` if the package is installed.
To exclude the slowest test, run `pytest -k "not slow" responsefun`.

Code style is enforced through `black` (formatting), `isort` (sorting import statements), and `ruff` (linting).

### Copyright

Copyright (c) 2023, The `responsefun` Developers


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
