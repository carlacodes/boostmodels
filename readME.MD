## Gradient-boosted regression and classification trees repo for behavioural animal data
work in progress for models I'm working on now to characterise main features of animal rxn time and correct response data across animal subjhects.
n.b. pymer4 DOES NOT WORK in python 3.10, downgrade to python 3.9 <br>

This is an implementation of lightGBM and SHAP to predict the probability of an animal making a false alarm, miss, or hit using gradient-boosted regression
and classification trees. 

To set up:
1. Install python 3.9
2. git clone this repository
3. cd into the repository
4. run `pip install -r requirements.txt`
5. run `python main.py`

<br> Please note, while I was able to run mixed effects models in python using pymer, I find the package incredibly clunky as
you are effectively calling R code from python. 
