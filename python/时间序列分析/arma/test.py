import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
dta = sm.datasets.sunspots.load_pandas().data[['SUNACTIVITY']]
dta.index = pd.date_range(start='1700', end='2009', freq='A')
print(type(dta.index))
exit()
res = sm.tsa.ARMA(dta, (3, 0)).fit()
fig, ax = plt.subplots()
ax = dta.loc['1950':].plot(ax=ax)
fig = res.plot_predict('1990', '2012', dynamic=True, ax=ax,
                        plot_insample=False)
plt.show()