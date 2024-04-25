## Gradient-boosted regression and classification trees repo for behavioural animal data

This is an implementation of lightGBM and SHAP to predict the probability of an animal making a false alarm, miss, or hit using gradient-boosted regression
and classification trees  (PLOS Computational Biology, in press, doi: https://doi.org/10.1371/journal.pcbi.1011985). This code assumes you have put your behavioural data into a pandas dataframe where each factor or variable of interest is represented as a column.

To set up:
1. Install python 3.9 (in this repository I used conda)
2. git clone this repository
3. cd into the repository
4. run `pip install -r requirements.txt`; the main requirements are the SHAP package, pandas, and lightGBM.
5. navigate to the models directory
6. run  the selected model, i.e. `runfalsealarmmodel.py` for the false alarm model

## Overview of models in this repository:
1. `runfalsealarmmodel.py` - classifier to predict whether there will be a false alarm during a catch trial
2. `reactiontimemodel.py` - regression model, predicts the reaction times of correct target trials
3. `reactiontimeforFAandtargmodel.py` - regression model, predicts the absolute reaction time relative to the trial start solely based on the words presented in the trial
4. `correctrespmissmodel.py` - predicts the probability of a miss/correct response during a target trial. <br>
`plotstatsacrossanimals.py` - plots general psychophysics metrics
## How to use this code
1. Put your data into a pandas dataframe where each column is a variable of interest
2. Change data paths as relevant and column names as relevant
3. Use optuna to fine tune your hyperparameters
4. Run the finalised model

