[![test artools](https://github.com/jbbutler/antarctic_AR_dataset/actions/workflows/test.yml/badge.svg)](https://github.com/jbbutler/antarctic_AR_dataset/actions/workflows/test.yml)

# Constructing Datasets of Antarctic Atmopsheric River Events: A New Catalog and Software Workflows

Welcome! This repository contains all of the source code, routines, notebooks used in the paper **insert paper here when complete**. To learn more about this work, feel free to check out the paper, or take a look at [the project website](https://jbbutler.github.io/antarctic_AR_dataset/), which contains a set of web-rendered notebooks that narrate the project with code and output.

This project also includes a python package called `artools`, which contains routines ranging from the unsupervised clustering algorithms used to produce the catalog of Antarctic ARs to functions to populate a tabular dataset with desired variables for each AR event. See the package documentation here (**insert link to package documentation**)..

## Installing `artools`

## Running the Routines

If you would like to reproduce this work, whether.. 

+ [A12D-01 Linking Antarctic Atmospheric River Characteristics with Their Landfalling Impacts](https://agu.confex.com/agu/agu25/meetingapp.cgi/Paper/1970828), [slides](https://zenodo.org/records/17926794)
+ [IN23A-06 Cloud-based Workflows for Antarctic Atmospheric Rivers: Successes and Challenges](https://agu.confex.com/agu/agu25/meetingapp.cgi/Paper/1971552), [slides](https://zenodo.org/records/17926811)

## Repo Contents
+ `environment.yml`: file specifying environment to run this workflow in
+ `dataset_construction.ipynb`: the notebook with the workflow implemented
+ `utils`: collection of helper modules to compute charcateristics and impacts of storms, given a streamed dataset and storm masks
+ `output`: directory containing output from the notebook
+ `data`: directory containing (1) a mask for the Antarctic Ice Sheet and (2) a dataset mapping lat/lon pixels to its area.
+ `catalog`: directory containing a subset of the full AR catalog we constructed (only first 250 storms, out of ~3000 total)

## Getting Started

A smaller-scale version of the full workflow we are developing can be found in `dataset_construction.ipynb`. If you'd like to try it out, feel free to clone this repository and run through the notebook: everything should be ready-to-run.

We implemented this workflow on [CryoCloud](https://cryointhecloud.com/), a cloud-hosted JupyterHub whose mission is to facilitate open, collaborative, and reproducible science. We recommend running through this notebook on CryoCloud to be able to replicate our exact workflow for yourself. However, if you'd like to run this notebook on some other computing service or JupyterHub, go ahead! Just make sure whichever platform you use is in the AWS `us-west-2` region, or else you won't be able to stream the MERRA-2 reanalysis data from the AWS S3 buckets.

If you need to create an account, doing to is very easy! See [here](https://book.cryointhecloud.com/getting-started).

Once you have an account and log in, follow these steps to start up a server you can use to run through the workflow!

<img width="1065" height="864" alt="cryo_walkthrough" src="https://github.com/user-attachments/assets/c771d5ea-e811-49f9-9722-42f8d2482489" />


