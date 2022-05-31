#K_MEANS

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

df = pd.read_csv(r"C:\Users\hp\PycharmProjects\pythonProject2\USArrests.csv", index_col=0)

df.head()

df.isnull().sum()

df.info()

df.describe().T


df.hist(figsize=(10,10));
plt.show()

kmeans=KMeans(n_clusters=4)
kmeans.get_params()
k_fit=kmeans.fit(df)

k_fit.n_clusters

k_fit.cluster_centers_
#There were four types of violence and we determined our cluster number as four.

k_fit.labels_
#We see which class the observation units are in.

#Visualization
#We will show clusters on two axes.For this reason, we chose two variables.
#Later we will do this with PCA.

k_means=KMeans(n_clusters=2).fit(df)
clusters=k_means.labels_
clusters
plt.scatter(df.iloc[:,0],df.iloc[:,1],c=clusters,s=50,cmap="viridis");
plt.show()
#We visualized two clusters.

centers=k_means.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c="black",s=200,alpha=0.5);
plt.show()
#If we want to look at the location of the centers of these two clusters.


plt.scatter(df.iloc[:,0],df.iloc[:,1],c=clusters,s=50,cmap="viridis");
plt.scatter(centers[:,0],centers[:,1],c="black",s=200,alpha=0.5);
plt.show()
#We can see clusters and their centers.
#Note:the last two visualization codes must be run together.



#Elbow Method

#Let's look at what our optimal number of clusters should be.

kmeans = KMeans()
sse_=[]

K=range(1,30)

for k in K:
    kmeans=KMeans(n_clusters=k).fit(df)
    sse_.append(kmeans.inertia_)

sse_
#distance of observation units from the center

plt.plot(K,sse_,"bx-")
plt.xlabel("sum of distance residual corresponding to k value")#differences
plt.title("elbow method for optimal number of clusters")
plt.show()

#Let's discuss the plot.We need to decide on the next point after the break we call the elbow.
#As the number of clusters increases, when the number of clusters is equal to the observation unit, the clustering process will have repeated each unit.
#That is, the number of clustering should always be less than the observation unit.Our number of observations was 50. As we move to the observation unit on the graph, we expect this error to decrease. Because there will be no errors.
#In other words, in a world where each observation unit itself is the center, there would be a situation where there would be no error.

###Alternative way for the Elbow Method###

#pip install yellowbrick

from yellowbrick.cluster import KElbowVisualizer

kmeans=KMeans()

visu=KElbowVisualizer(kmeans,k=(2,20))
#The first argument is the kmeans object we created, and the second is the set of numbers that I gave intervals to try.
visu.fit(df)
visu.show()

#What has changed?He marked the point we looked at manually on the graph itself. The dashed lines.
#elbow at k=6, It gave the value he suggested.6 is the most optimal number, we should divide our dataset into 6 clusters
#we need to create our final kmeans model


#We will build our model with the k we have determined according to the elbow.
kmeans=KMeans(n_clusters=4).fit(df)
kmeans
kmeans.get_params()
#We built the model, but who belongs to which state?
#We will combine the cluster numbers with the results we have and create a dataframe.
kumeler=kmeans.labels_
pd.DataFrame({"states":df.index,"clusters":clusters})

df["cluster_no"]=clusters

####Hierarchical clustering ####

from scipy.cluster.hierarchy import linkage
hc_complete=linkage(df,"complete")
hc_average=linkage(df,"average")
#We will use these two objects to construct the dendrogram.

from scipy.cluster.hierarchy import dendrogram
#Now let's see via the image.

plt.figure(figsize=(20,10))
plt.title("hierarchical_clustering_dendrogram")
plt.xlabel("observation_units")
plt.ylabel("distances")
dendrogram(hc_complete,leaf_font_size=10,);
plt.show()



plt.figure(figsize=(15,10))
plt.title("hierarchical_clustering_dendrogram")
plt.xlabel("observation_units")
plt.ylabel("distances")
dendrogram(hc_complete,truncate_mode="lastp",p=4,show_contracted=True,leaf_font_size=10,);
plt.show()
#The result of this plot has been divided into 4 clusters. The numbers of observation units related to these 4 clusters have also been given.(2)(14)(14)(20)
#truncate_mode="lastp" show lastly p pieces.
#leaf_font_size=10 for font display
#show_contracted=True When I want it to bring the information of how many elements there are when the clusters are made.

plt.figure(figsize=(20,10))
plt.title("hierarchical_clustering_dendrogram")
plt.xlabel("observation_units")
plt.ylabel("distances")
dendrogram(hc_average,leaf_font_size=10,);
plt.show()
#again same process, difference is not hc complete, it is given hc_average.



###PCA###
df=pd.read_csv(r"C:\Users\hp\PycharmProjects\pythonProject2\USArrests.csv")
df.dropna(inplace=True)
df=df._get_numeric_data()
df.head()

from sklearn.preprocessing import StandardScaler
df=StandardScaler().fit_transform(df)
df[0:5,0:5]

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca_fit=pca.fit_transform(df)
#now reduced to two components

comp_df=pd.DataFrame(data=pca_fit,columns=["firscomponent","secondcomponent"])
#in this way, it has been reduced to two variables.we can now represent with two variables.

#Actually, the work is over, but let's understand the theoretical logic of the calculations on the back a little more...

pca.explained_variance_ratio_
#array([0.45245466, 0.24246801])
#45 percent of the variables in the first component data set were explained
#Similarly, 24 percent of the variability in the data set could be explained.
#If these two add up, approximately will be %67-68 70 explained with two variables.So they can explain the variability in the dataset(2 dimensions)
#If I think that there are 100 variables, we can change the value of 100 variables to 2 or 3 variables.
#As can be seen from the rate of information loss, however, the variability was preserved.

pca.components_
#we access components which are all components but there are 2 components.

pca.components_[1]
#We selected one components.
#we had decided 2 components with n_compenents.



#now let's put other component numbers in the same study and examine how the explained variance changes
#For example, we might want to decide on the number of components with this.let's look at the explained variance.


#We want to decide on the optimum number of components.
pca=PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_));
plt.xlabel("numberOFcomponent")
plt.ylabel("cumulative_variance_ratio")
plt.show()
#In the previous process, that is, when setting up for the first time, we gave 2 variables and saw 2 as output./explained_varianca_ratio
#If we don't specify a component count as now, as many as the variable number of components will be created.
#In this case, if we take the cumulative sums of the components that consist of so many variables,
#when we come to the 2nd component, we can learn how much of the variability in the data set is explained when it is considered together with the 1st.

#As a result of the plot,we observe that as the number of components increases, the variance in the data set(the information the data set contains) goes to explain all the increased information in the plot.
#If each of the variables is a component, an information from the data set will not be lost.

#NOTE:actually 1.component starting with 0 in the plot. But starting from zero and increases one by one (component numbers, I mean x axis)


pca.explained_variance_ratio_


#assume that decided to 3 components ,final model with 3 components.
pca=PCA(n_components=3)
pca_fit=pca.fit_transform(df)
pca.explained_variance_ratio_
#We represented about 80 percent  with 3 components.








