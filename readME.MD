## Gradient-boosted regression and classification trees repo for behavioural animal data

This is an implementation of lightGBM and SHAP to predict the probability of an animal making a false alarm, miss, or hit using gradient-boosted regression
and classification trees. This code assumes you have put your behavioural data into a pandas dataframe where each factor or variable of interest is represented as a column.

To set up:
1. Install python 3.9
2. git clone this repository
3. cd into the repository
4. run `pip install -r requirements.txt`
5. navigate to the models directory
6. run  the selected model, i.e. `runfalsealarmmodel.py` for the false alarm model

