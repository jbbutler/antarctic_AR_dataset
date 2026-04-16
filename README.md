[![test artools](https://github.com/jbbutler/antarctic_AR_dataset/actions/workflows/test.yml/badge.svg)](https://github.com/jbbutler/antarctic_AR_dataset/actions/workflows/test.yml)

# Reproducible Workflows to Construct Event-Based Datasets of Geophysical Phenomena from Threshold-Based Products: an Illustration with Antarctic Atmospheric Rivers

Welcome! This repository contains all of the source code, routines, notebooks used in the paper **insert paper here when complete**. To learn more about this work, feel free to check out the paper, or take a look at [the project website](https://jbbutler.github.io/antarctic_AR_dataset/), which contains a set of web-rendered notebooks with project code and output.

In addition to the main paper and website, this project consists of 2 other deliverables:
1. A new storm-by-storm catalog of Antarctic AR events
2. A Python package called `artools`, which contains our implementation of our clustering algorithm as well as useful tools and routines to manage a generic catalog of event

## Accessing the AR Catalog

The official Antarctic AR catalog can be accessed from a [HuggingFace repository](https://huggingface.co/datasets/butlerjorg/antarctic_AR_catalogs).

## Installing `artools`

To install `artools`, run the following command:

```
pip install artools
```

## Repo Content
+ `artools`: the `artools` package, with associated modules and classes
+ `input_data`: ...
+ `notebooks`: 
+ `output`:
+ `scripts`:
+ `environment.yml`:
+ `index.md`:
+ `myst.yml`:
+ `pyproject.toml`:

## Getting Started

You may find yourself wanting to either reproduce this work, or try out the workflows for your own problem! There are two ways in which this work can be reproduced: reproducing the clustering output, and reproducing the dataset construction. For the former, ... for the latter...

We implemented this workflow on [CryoCloud](https://cryointhecloud.com/), a cloud-hosted JupyterHub whose mission is to facilitate open, collaborative, and reproducible science. We recommend running through this notebook on CryoCloud to be able to replicate our exact workflow for yourself. However, if you'd like to run this notebook on some other computing service or JupyterHub, go ahead! Just make sure whichever platform you use is in the AWS `us-west-2` region, or else you won't be able to stream the MERRA-2 reanalysis data from the AWS S3 buckets.

If you need to create an account, doing to is very easy! See [here](https://book.cryointhecloud.com/getting-started).

Once you have an account and log in, follow these steps to start up a server you can use to run through the workflow!

<img width="1065" height="864" alt="cryo_walkthrough" src="https://github.com/user-attachments/assets/c771d5ea-e811-49f9-9722-42f8d2482489" />

## Previous Presentations

Here's a list of prior presentations in which this work has been mentioned.

+ [A12D-01 Linking Antarctic Atmospheric River Characteristics with Their Landfalling Impacts](https://agu.confex.com/agu/agu25/meetingapp.cgi/Paper/1970828), [slides](https://zenodo.org/records/17926794)
+ [IN23A-06 Cloud-based Workflows for Antarctic Atmospheric Rivers: Successes and Challenges](https://agu.confex.com/agu/agu25/meetingapp.cgi/Paper/1971552), [slides](https://zenodo.org/records/17926811)





