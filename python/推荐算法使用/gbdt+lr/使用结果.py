import pandas as pd
import numpy as np
import joblib
import time
import configparser
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder

norm=joblib.load(base_dir+'/归一化.pkl')
joblib.dump(norm,types+'/types_onehot.pkl')