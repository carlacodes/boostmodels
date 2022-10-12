## Mixed effect models repo for behavioural animal data
work in progress for models I'm working on now to characterise main features of animal rxn time and correct response data across animal subjhects.
n.b. pymer4 DOES NOT WORK in python 3.10, downgrade to python 3.9 <br>
I find that using pymer4 gives the most complex models, but also the most interpretable results. <br>
I'm using the pymer4 package in python to fit mixed effect models to behavioural data, but statsmodels and pysr3 also works,
but I can't figure out how to mark categorical variables. <br>
to use pymer4, you need to have installed R > 3.6, for some reason 4.2. did not work with my conda env so I had to downgrade to R 4.1 <br> 
[]: # Language: Python
[]: # Path: trackBehaviourAnalysis_mod.py
