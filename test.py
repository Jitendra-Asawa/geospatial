'''
Data set image 1
4 random points selected to make groups are:
Water: (89,70,77)
Vegetation: (99,81,238)
Hills: (97,91,118)
Built-Up: (140,149,170)
Color Composition for representation: False Color Composite (FCC)
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
import pandas as pd
from pandas import *
from mpl_toolkits.mplot3d import Axes3D

style.use('ggplot')
#KNN 
def k_nearest_neighbors(data, predict, k=5):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
        
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            #print(np.array(features),'  ', np.array(predict), ' ' , euclidean_distance,'\n')
            distances.append([euclidean_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

dataset = {'k':[89,70,77], 'r':[97,91,118], 'g':[99,81,238], 'c':[140,149,170]}
new_features = [89,95,115]  # test point introduced while building code

# Reading and converting data into required format
df= pd.read_csv('C:/Users/Pooja/dataset.csv') 
print('Size : ', df.size,'\n')
#convert data into list of list where each inner list is made of row
df= df.astype(int).values.tolist()
print('Data Frame : ' ,df, '\n')

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')

# Calculating the group to which a point belongs 
j=0
for i in df:
    result = k_nearest_neighbors(dataset, i)
    print(j,'\n',result)
    ax.scatter(df[j][0], df[j][1], df[j][2],color= result ,marker = 'o')
    j = j + 1

#access value of a point rather that individual element in list value
for i in dataset:
    #print(i[0],' ',i[1],' ',i[2],'\n')
        ax.scatter(dataset[i][0],dataset[i][1],dataset[i][2],color= i ,marker = 'o')
    
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
