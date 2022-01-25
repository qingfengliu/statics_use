import pandas as pd
from sklearn.datasets import make_classification

#主要是用于非独热转换.
#Yes P:P-ESPN:1 A:A-Nike:1 G:G-Male:1.
class FFMFormatPandas:
    def __init__(self):
        self.field_index_ = None
        self.feature_index_ = None
        self.y = None

    def fit(self, df, y=None):
        self.y = y
        temp=[x[0] for x in df.columns.str.split(':')]
        temp.remove(y.name)
        self.field_index_ = dict(zip(temp,range(1,len(temp)+1)))
        print(self.field_index_)

        if self.feature_index_ is not None:
            last_idx = max(list(self.feature_index_.values()))

        if self.feature_index_ is None:
            self.feature_index_ = dict()
            last_idx = 0

        for col in df.columns:
            if col==self.y.name:
                continue
            if col.split(':')[1] == 'cate':
                vals = df[col].unique()
                for val in vals:
                    if pd.isnull(val):
                        continue
                    name = '{}_{}'.format(col, val)

                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            elif col.split(':')[1] == 'con':
                if col not in self.feature_index_:
                    self.feature_index_[col] = last_idx
                    last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self,idx, row):
        ffm = []
        if self.y is None:
            ffm.append(str(0))
        else:
            ffm.append(str(self.y.iloc[idx]))

        for col, val in row.loc[row.index != self.y.name].to_dict().items():
            if len(col.split(':'))>1:
                if col.split(':')[1] == 'cate':
                    val = int(val)
                    name = '{}_{}'.format(col, val)
                    ffm.append('{}:{}:1'.format(self.field_index_[col.split(':')[0]], self.feature_index_[name]))
                elif col.split(':')[1] == 'con':
                    ffm.append('{}:{}:{}'.format(self.field_index_[col.split(':')[0]], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        return pd.Series({idx: self.transform_row_(idx,row) for idx, row in df.iterrows()})

########################### Lets build some data and test ############################

train, y = make_classification(n_samples=100, n_features=5, n_informative=2, n_redundant=2, n_classes=2, random_state=42)

train=pd.DataFrame(train, columns=['a1:cate','a2:cate','a3:cate','a4:con','a5:con'])
train['a1:cate'] = train['a1:cate'].astype(int)
train['a2:cate'] = train['a2:cate'].astype(int)
train['a3:cate'] = train['a3:cate'].astype(int)
train['clicked'] = y
print(train.head())

ffm_train = FFMFormatPandas()
ffm_train.fit(train,train['clicked'])
print(ffm_train.feature_index_)
ffm_train_data=ffm_train.transform(train)


ffm_train_data.to_csv('a.csv',index=False,header=False)
