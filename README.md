# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Choose the number of clusters (K): Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.
2. Initialize cluster centroids: Randomly select K data points from your dataset as the initial centroids of the clusters.
3. Assign data points to clusters: Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically done using Euclidean distance, but other distance metrics can also be used.
4. Update cluster centroids: Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.
5. Repeat steps 3 and 4: Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.
6. Evaluate the clustering results: Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.
7. Select the best clustering solution: If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements



## Program:

Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Gokul ,
RegisterNumber:  212221220013
```py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters = i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow matter")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segmets")

```


## Output:
### data.head() :
![image](https://github.com/babavoss05/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/103019882/8f7c6306-1532-4358-9bcd-18b7d7b22d25)

### data.info() :
![image](https://github.com/babavoss05/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/103019882/576d685d-40ff-496b-95e6-408d1954d03c)


### Null Values :
![image](https://github.com/babavoss05/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/103019882/7deb121f-e8a6-4b3c-846f-bbc6393faa07)

### Elbow Graph :
![image](https://github.com/babavoss05/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/103019882/2725e886-2fe8-4327-9c71-bc0d68612cab)

### K-Means Cluster Formation :
![image](https://github.com/babavoss05/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/103019882/a4606964-5df8-4772-9851-df18730e094f)

### Predicted Value :
![image](https://github.com/babavoss05/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/103019882/3f2b769b-8055-4c03-bff9-ae24c7601a6b)

### Final Graph :
![image](https://github.com/babavoss05/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/103019882/0e86030b-f450-4601-9342-2d6ddbaf1f23)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
