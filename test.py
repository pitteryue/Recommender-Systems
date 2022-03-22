import numpy as np
import pandas as pd
import copy

arr = np.array([[1,0,1],[1,1,1],[0,0,1]])
R= pd.DataFrame(arr)
degree_R = copy.deepcopy(R)

for i in range(len(R.columns)):
    item_degree = R[i].value_counts().loc[1]
    degree_R.iloc[:,i] = degree_R.iloc[:,i] / item_degree

print(R)
print(degree_R)