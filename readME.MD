## Mixed effect models repo for behavioural animal data
work in progress for models I'm working on now to characterise main features of animal rxn time and correct response data across animal subjhects.
n.b. pymer4 DOES NOT WORK in python 3.10, downgrade to python 3.9 <br>
I find that using pymer4 gives the most complex models, but also the most interpretable results. <br>
I'm using the pymer4 package in python to fit mixed effect models to behavioural data, but statsmodels and pysr3 also works,
but I can't figure out how to mark categorical variables in these two packages. For pymer4, I recommend using the OLD optimizer. <br>
to use pymer4, you need to have installed R > 3.6, for some reason 4.2. did not work with my conda env so I had to downgrade to R 4.1 <br> 
For model selection (i.e. feature or variable elimination), I used BIC which you are aiming to MINIMIZE.<br>
Stepwise, or recursive feature elimination does not work well for mixed models. For the R^2 metric, I'm using the Nakagawa 2012 measure of the R2 coefficient for mixed models. 
Kind of controversial because it is a mixed model but it's still a solid metric that works by considering both the conditional and marginal variance. 
This uses the easystats nakagwa_r2 function and it is one line of code to import it into Python <br>
[]: # Language: Python
[]: # Path: trackBehaviourAnalysis_modv3simplified.py
