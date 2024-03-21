# Example project
It's a example repository intended to demonstrate how to create a geospatial project with Python using good software development practices. It's also intended to be used as a template for future projects.

This repository is organized as a Python package and additional code for running experiments.

## Dependencies management and package installation
The following comand can be used to recreate the conda enviroment with all the dependencies needed to run the code in this repository.
```
conda env create -f environment.yml
```
After creating the new enviroment (or you can use an existing one) you need to activate it and install the package in development mode. To do so, from the repository root, run the command below. It will install the package in development mode, so you can make changes to the code and test it without the need to reinstall the package.
```
pip install -e .
```
You can also install the package directly from GitHub using the following command:
```
pip install git+https://github.com/lbferreira/geospatial_project_example
```

## Project structure
This repository contains the following structure:
```
project_root_folder/
└───data/
└───docs/
└───notebooks/
└───src/
│   └───mypackage/
│       │  __init__.py
│       │  mymodule.py
│       │  mymodule2.py
|       │   └───mysubpackage/
|       │       │  __init__.py
|       │       │  mysubmodule.py
|  .gitignore
│  environment.yml
│  pyproject.toml
│  README.md
```

- `data/`: Folder to store data used in the project. However, it's not recommended to store large files in the repository. Overall, you just keep the code in the repository and small files.
- `docs/`: Folder to store documentation files, such PDFs, images, documents, etc.
- `notebooks/`: Folder to store Jupyter notebooks. It's recommended to use notebooks only for exploratory analysis or experiments. The "heavy" code should be in the package.
- `src/mypackage/`: Folder to store the package code. You can also use subpackages.
- `.gitignore`: File to specify which files and folders should be ignored by Git.
- `environment.yml`: File to specify the dependencies of the project based on Conda. It can be used to recreate the conda environment.
- `pyproject.toml`: File to specify the project configuration, such as the package name, version, and dependencies.
- `README.md`: File to describe the repository.

## Code examples
In the folder [notebooks](./notebooks/) there are three Jupyter notebooks with examples on how to improve code quality. It's not intended to be the best way to do the tasks provided, but it present some useful tips to improve code quality.
- [example.ipynb](./notebooks/example.ipynb): Example of a code with bad practices.
- [example_refactored.ipynb](./notebooks/example_refactored.ipynb): This a proposed refactored version of the code in the previous notebook.
- [s2scan.ipynb](./notebooks/s2scan.ipynb): Example of a code with good practices.

## Aditional information
This material was created as a part of an internal training for the members of the lab [GCER](https://www.gcerlab.com/)
![](./docs/gcer_logo.png)