# Example project
It's a example repository intended to demonstrate how to create a Python project using good software development practices. It's also intended to be used as a template for future projects.

## Description
This repository is organized as a Python package and additional code for running experiments. The package is organized in...

## Dependencies management
The following comand can be used to recreate the conda enviroment with all the dependencies needed to run the code in this repository.
```
conda env create -f environment.yml
```
After creating the new enviroment (or you can use an existing one) you need to activate it and install the package in development mode. To do so, from the repository root, run the command below. It will install the package in development mode, so you can make changes to the code and test it without the need to reinstall the package.
```
pip install -e .
```
