import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
                                                PoissonBayesMixedGLM)
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime

np.random.seed(8767)
n = 200
m = 20
data = pd.DataFrame({"Year": np.random.uniform(0, 1, n),
                     "Village": np.random.randint(0, m, n)})
data['year_cen'] = data['Year'] - data.Year.mean()

# Binomial outcome
lpr = np.random.normal(size=m)[data.Village]
lpr += np.random.normal(size=m)[data.Village] * data.year_cen
y = (np.random.uniform(size=n) < 1 / (1 + np.exp(-lpr)))
data["y"] = y.astype(int)

# These lines should agree with the example in the class docstring.
random = {"a": '0 + C(Village)'}

print(data)
model = BinomialBayesMixedGLM.from_formula(
    'y ~ year_cen', random, data)
result = model.fit_vb()