import numpy as np
import statsmodels.genmod.generalized_estimating_equations as gee
import pandas as pd
#1.有序gee
np.random.seed(434)
n = 40
y = np.random.randint(0, 3, n)
groups = np.arange(n)
x1 = np.random.normal(size=n)
x2 = np.random.normal(size=n)

df = pd.DataFrame({"y": y, "groups": groups, "x1": x1, "x2": x2})

model = gee.OrdinalGEE.from_formula("y ~ 0 + x1 + x2", groups, data=df)
model.fit()

#2.多元GEE
np.random.seed(434)
n = 40
y = np.random.randint(0, 3, n)
groups = np.arange(n)
x1 = np.random.normal(size=n)
x2 = np.random.normal(size=n)

df = pd.DataFrame({"y": y, "groups": groups, "x1": x1, "x2": x2})

model = gee.OrdinalGEE.from_formula("y ~ 0 + x1 + x2", groups, data=df)
model.fit()
