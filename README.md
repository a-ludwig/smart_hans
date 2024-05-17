# smart_hans
This repository contains all components that are needed to collect data, train, optimize and exhibit Smart Hans. 


## Setup

### Software
Install vlc player 64-bit (https://www.videolan.org)
Install Conda https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html
Set up an environment using "smart_hans_2024.yml". Open a conda terminal and navigate to this base directory. then create the environment by using this command: 

conda create --name smart_hans --file smart_hans_2024.yml


### Hardware

Check the setup instruction in the [!Instructions directory](!Instruction) and set up hardware accordingly.

## Running Smart Hans

### Installation_V1 in branch main (lab30)
To run Hans in an exhibition setup run [main.py](main.py). Hans will then work according to the Instructions given in [Hansi_Interaktionsanleitung](Hansi_Interaktionsanleitung.md).

The model in the export directory will be loaded for making predictions.

### Installation_V2 in branch feeback (MCBW)
**On RasPi** <br>
Run [raspi/feedback.py](https://github.com/a-ludwig/smart_hans/blob/feedback/raspi/feedback.py) on your raspberry pi

**On PC** <br>
To test the connection to the RasPi run [raspi/test_socket.py](https://github.com/a-ludwig/smart_hans/blob/feedback/test_socket.py) <br>
To run Hans in an exhibition setup run [main.py](main.py). Hans will then work according to the Instructions given in "Hansi_Interaktionsanleitung".

## Training Smart Hans

### Collecting Data

[Datensammeln](datensammeln) contains all that is needed to collect data. [Videocapture.py](datensammeln/videocapture.py) runs an application that records and plays videos in a specific format. Running it will prompt a few markers to be set.

1. Acronym for Gender (m/w/d), Height(k/n/g), Face Blocked (y/n). These are by no means accurate measures and so far they have not been used other than for this step.
2. Number the person will be thinking of.

When all markers are set, Smart Hans will start tapping. The Horse will tap for a specific amount of taps which is currently set to 15. People standing in front of Hans were told that the horse will stop counting at the number they are thinking of. 

### Preprocessing Data

To preprocess Videos that have  been collected use [hansi_preprocessing.py](Machine_learning/hansi_preprocessing.py). It takes to arguments "path" and "out". Path is the folder containing your prerecorded videos. "out" is your desired output path. 

### Machine Learning

To work with the preprocessed Data use the corresponding notebooks.

[tsai_with_real_data_univariate/multivariate.ipynb](Machine_learning/tsai_optuna_optimize_with_real_data_univariate.ipynb) can be used for training a single model.
tsai_compare_... can be used to train and compare multiple models.
tsai_optuna... can be used to optimize a univariate model.

[vis_test.py](Machine_learning/vis_test.py) can be used to visualize the data.

## What else is there?

### Installation Export

Installation Export contains exports that are being made when the Installation is exhibited. It contains unlabeled data of when Hans guessed any number when interacting with visitors. This might be used in the future for unsupervised training.




