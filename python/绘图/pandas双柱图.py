import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

name_list=["A","B","C"]
pair_list=[[1,1],[2,3],[4,5]]
pair_list=np.array(pair_list)
img_df=pd.DataFrame(pair_list,index=name_list,columns=["col 1","col 2"])
img_df.plot(kind="bar",rot=0)
print(img_df)
# plt.savefig("bar.jpg")
plt.show()

