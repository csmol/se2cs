# STABILIZER + TIMELIME

Code repository for library stabilizer-timelime - https://pypi.org/project/stabilizer-timelime/

### Description

This project is a package version of [STABILIZER](https://arxiv.org/abs/1911.04250) and [TimeLIME]().

Stabilizer generates the hierarchical clustering of the projects and develops a predictive model for the goal for every project. It then promotes the best model at each level to the level above. At level 0, we get the model/project which generalizes of all of the projects.

TimeLime on the other hand, suggests plan to improve the goal (increase/decrease). It only suggests the plans that it has seen in the past.

### How to build the package?

To build the package we use poetry as build-backend. Install poetry using pip - <br>
`pip install poetry` or `pip3 install poetry`

In root directory (directory that contains `pyproject.toml`) run the build command - <br>
`python3 -m build` <br>
This command will generate a wheel (a pre-built binary package) in `dist/` directory <br>


### How to install this package?

To install the latest version run - <br>
`pip install stabilizer-timelime` or `pip3 install stabilizer-timelime`


To install custom built wheel - <br>
`pip install <path-to-wheel-file>` or `pip3 install <path-to-wheel-file>`