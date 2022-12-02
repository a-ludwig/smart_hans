# smart_hans
This repository contains all components that are needed to collect data, train, optimize and exhibit Smart Hans. 


## Setup

### Software

Install requirements in requirements.txt. Make sure to have a working CUDA installation. Smart Hans worked fine with CUDA 11.6.

### Hardware

Check the setup instruction in the !Instructions directory and set up hardware accordingly.

## Running Smart Hans

To run Hans in an exhibition setup run "main.py". Hans will then work according to the Instructions given in "Hansi_Interaktionsanleitung".

The model in the export directory will be loaded for making predictions.



## Training Smart Hans

### Collecting Data

Datensammeln contains all that is needed to collect data. "Videocapture.py" runs an application that records and plays videos in a specific format. Running it will prompt a few markers to be set.

1. Acronym for Gender (m/w/d), Height(k/n/g), Face Blocked (y/n). These are by no means accurate measures and so far they have not been used other than for this step.
2. Number the person will be thinking of.

When all markers are set, Smart Hans will start tapping. The Horse will tap for a specific amount of taps which is currently set to 15. People standing in front of Hans were told that the horse will stop counting at the number they are thinking of. 

### Preprocessing Data

To preprocess Videos that have  been collected use "hansi_preprocessing.py". It takes to arguments "path" and "out". Path is the folder containing your prerecorded videos. "out" is your desired output path. 

### Machine Learning

To work with the preprocessed Data use the corresponding notebooks.

tsai_with_real_data_univariate/multivariate.ipynb can be used for training a single model.
tsai_compare_... can be used to train and compare multiple models.
tsai_optuna... can be used to optimize a univariate model.

vis_test.py can be used to visualize the data.

## What else is there?

### Installation Export

Installation Export contains exports that are being made when the Installation is exhibited. It contains unlabeled data of when Hans guessed any number when interacting with visitors. This might be used in the future for unsupervised training.




