# Example Machine Learning Project Workflow
This repository uses a series of Jupyter notebooks to demonstrate what a typical Machine Learning (ML) project workflow looks like when using Python and the SciPy stack: Pandas, Numpy and SciKit-Learn. 

The ML task is to predict median house values in a given region of California - i.e. it is a regression task. It is adapted from the example project given in Chapter 2 of 'Hands on Machine Learning with SciKit-Learn and TensorFlow', by Aurelien Geron (a great read on current approaches to ML, that I recommend).

I have split the ML project workflow into the following notebooks:

1. Data Retrieval
2. Data Exploration
3. Data Preparation
4. Data Modelling

The outputs of this project are a data preparation pipeline and machine learning model that can be loaded from disk and used in a 'production like environment' - e.g. one could construct a simple REST API written using Flask and deploy it in a Docker container, that could then be scaled-up using something like AWS's ElasticBeanstalk or Elastic Container Services.

Also included in this repository in a simple python module - `custom_transformers.py` - that contains a couple of classes that define custom transformations used in the final data preparation pipeline.