[![test artools](https://github.com/jbbutler/antarctic_AR_dataset/actions/workflows/test.yml/badge.svg)](https://github.com/jbbutler/antarctic_AR_dataset/actions/workflows/test.yml)

# Reproducible Workflows to Construct Event-Based Datasets of Geophysical Phenomena from Threshold-Based Products: an Illustration with Antarctic Atmospheric Rivers

Welcome! This repository contains all of the source code, routines, and notebooks used in the associated paper. To learn more about this work, feel free to check out the paper, or take a look at [the project website](https://jbbutler.github.io/antarctic_AR_dataset/), which contains a set of web-rendered notebooks with project code and output.

In addition to the main paper and website, this project consists of 2 products:
1. A new storm-by-storm catalog of Antarctic AR events, based on the Wille (2021) MERRA-2 vIVT threshold catalog
2. A Python package called `artools`, which contains our implementation of our clustering algorithm as well as useful routines to manage a generic catalog of events (*TODO: add link to documentation here*)

## Getting Started with the AR Catalog

The official Antarctic AR catalog can be accessed from a [HuggingFace repository](https://huggingface.co/datasets/butlerjorg/antarctic_AR_catalogs). You can either download it directly, or use `artools.loading_utils.load_catalog(<filename>)` to load it into a notebook or script directly from HuggingFace.

## Getting Started with `artools`

`artools` can be installed via pip:
```
pip install artools
```

This package is used extensively throughout this work. If you would like more details, see *the documentation*. 

## Repo Content
+ `artools`: the `artools` package, with associated modules and classes
+ `input_data`: a directory for data products not produced by this work but used in some capacity
+ `notebooks`: notebooks detailing figures and analyses of the project, including the dataset construction workflows
+ `output`: a directory to which output produced by notebooks and scripts are saved
+ `scripts`: scripts used to run the clustering algorithm on the Wille (2021) catalog with various hyperparameters
+ `environment.yml`: package specifications for a conda environment/image to run this repo's code
+ `index.md`: the landing page for the MyST website
+ `myst.yml`: project-level specifications for the MyST website
+ `pyproject.toml`: configuration file for the `artools` package

## Running the Code

You may find yourself wanting to either reproduce this work, or try out the workflows for your own problem!

If you would like to run our clustering algorithm for your own gridded threshold dataset, you can use the `artools.ST_DBSCAN` class and the class's `fit()` method. Depending on how large your dataset is and how many pixels have positive binary labels, you may consider leaving the job to run for a few hours on a cluster. If you would like to reproduce all of our catalogs for the various combinations of hyperparameters examined, you can call up the `clustering.py` script in the `scripts` directory in your own SLURM script. We provide our own SLURM scripts as well for reference, but these are particular to submitting jobs on the UC Berkeley Statistical Computing Facility cluster, which is what was used in this analysis.

If you would like to run the dataset streaming workflow, or reproduce the output of any of our notebooks, you should simply be able to run the notebooks. Since the workflows in these notebooks are very lightweight, we recommend you use [CryoCloud](https://cryointhecloud.com/), a cloud-hosted JupyterHub whose mission is to facilitate open, collaborative, and reproducible science. If you need to create an account, doing to is very easy! See [here](https://book.cryointhecloud.com/getting-started).

Once you have an account and log in, you can spin up a CPU server. Select 'Build your own image' in the 'Environment' dropdown, enter the repository which has the `environment.yml` (either this one or your own clone), click 'Build image', and wait until it's built! Finally, choose your server size and then click 'Start'. You're ready to run!



## Previous Presentations

Here's a list of prior presentations in which this work has been mentioned.

+ [(AGU25) A12D-01 Linking Antarctic Atmospheric River Characteristics with Their Landfalling Impacts](https://agu.confex.com/agu/agu25/meetingapp.cgi/Paper/1970828), [slides](https://zenodo.org/records/17926794)
+ [(AGU25) IN23A-06 Cloud-based Workflows for Antarctic Atmospheric Rivers: Successes and Challenges](https://agu.confex.com/agu/agu25/meetingapp.cgi/Paper/1971552), [slides](https://zenodo.org/records/17926811)





