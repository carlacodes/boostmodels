from pysr3.lme.models import L1LmeModelSR3
from pysr3.lme.problems import LMEProblem, LMEStratifiedShuffleSplit
import numpy as np
from pysr3.linear.models import LinearL1ModelSR3
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform

import numpy as np

from pysr3.linear.problems import LinearProblem

# Create a sample dataset
seed = 42
num_objects = 300
num_features = 500
np.random.seed(seed)
# create a vector of true model_name's coefficients
true_x = np.random.choice(2, size=num_features, p=np.array([0.9, 0.1]))
# create sample data
a = 10 * np.random.randn(num_objects, num_features)
b = a.dot(true_x) + np.random.randn(num_objects)

print(f"The dataset has {a.shape[0]} objects and {a.shape[1]} features; \n"
      f"The vector of true parameters contains {sum(true_x != 0)} non-zero elements out of {num_features}.")
problem, true_parameters = LMEProblem.generate(
    groups_sizes=[10] * 6,  # 6 groups, 10 objects each
    features_labels=["fixed+random"] * 20,  # 20 features, each one having both fixed and random components
    beta=np.array([0, 1] * 10),  # True beta (fixed effects) has every other coefficient active
    gamma=np.array([0, 0, 0, 1] * 5),  # True gamma (variances of random effects) has every fourth coefficient active
    obs_var=0.1  # The errors have standard errors of sqrt(0.1) ~= 0.33

)

# LMEProblem provides a very convenient representation
# of the problem. See the documentation for more details.

# It also can be converted to a more familiar representation
x, y, columns_labels = problem.to_x_y()
# columns_labels describe the roles of the columns in x:
# fixed effect, random effect, or both of those, as well as
# fixed effect, random effect, or both of those, as well as
# We use SR3-empowered LASSO model_name, but many other popular models are also available.
# See the glossary of models for more details.
model = L1LmeModelSR3()
# x is  a long array of dependent VARS
# y is the independent VAR for your prediction

# We're going to select features by varying the strength of the prior
# and choosing the model_name that yields the best information criterion
# on the validation set.
params = {
    "lam": loguniform(1e-3, 1e3)
}
# We use standard functionality of sklearn to perform grid-search.
selector = RandomizedSearchCV(estimator=model,
                              param_distributions=params,
                              n_iter=10,  # number of points from parameters space to sample
                              # the class below implements CV-splits for LME models
                              cv=LMEStratifiedShuffleSplit(n_splits=2, test_size=0.5,
                                                           random_state=seed,
                                                           columns_labels=columns_labels),
                              # The function below will evaluate the information criterion
                              # on the test-sets during cross-validation.
                              # We use IC from Muller2018, but other options (AIC, BIC) are also available
                              scoring=lambda clf, x, y: -clf.get_information_criterion(x,
                                                                                       y,
                                                                                       columns_labels=columns_labels,
                                                                                       ic="muller_ic"),
                              random_state=seed,
                              n_jobs=20
                              )
selector.fit(x, y, columns_labels=columns_labels)
best_model = selector.best_estimator_

maybe_beta = best_model.coef_["beta"]
maybe_gamma = best_model.coef_["gamma"]
#returning a binary array of 0 1 true false to get the actually relevant values?? -- yes as in the sk learn function y_true is the first argument and y_pred is the second argument
ftn, ffp, ffn, ftp = confusion_matrix(true_parameters["beta"], abs(maybe_beta) > 1e-2).ravel()
rtn, rfp, rfn, rtp = confusion_matrix(true_parameters["gamma"], abs(maybe_gamma) > 1e-2).ravel()
#TODO: does this package have the AIC and BIC
print(
    f"The model_name found {ftp} out of {ftp + ffn} correct fixed features, and also chose {ffp} out of {ftn + ffn} extra irrelevant fixed features. \n"
    f"It also identified {rtp} out of {rtp + rfn} random effects correctly, and got {rfp} out of {rtn + rfn} non-present random effects. \n"
    f"The best sparsity parameter is {selector.best_params_}")