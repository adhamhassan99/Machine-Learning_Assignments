import math
from random import randint
import numpy as np
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

my_data = pd.read_csv("C:\\Users\Adham\\Documents\\Machine Learning\\Machine-Learning_Assignments\\Assignment 1\\Customer data.csv")
# print(my_data.head())

def GUC_Distance ( Cluster_Centroids, Data_points, Distance_Type ):
## write code here for the Distance function here #
    
    Cluster_Distance=[]
    if Distance_Type=="Ecluidian distance":
        for point in Data_points:
            row=[]
            for centroid in Cluster_Centroids:
                x=(point-centroid)^2
                x=math.sqrt(Cluster_Distance)
            row.append(x)
            Cluster_Distance.append(row)
                

    return Cluster_Distance 

def GUC_Kmean ( Data_points, Number_of_Clusters,  Distance_Type ):
       # write code for intial cluster heads here
       clusters=[]
       for i in range(Number_of_Clusters):
          clusters.append(Data_points[randint(0, len(Data_points))])
        
       # write your your loop
       distances=GUC_Distance(clusters,Data_points,"Ecluidian distance")
       Cluster_Metric=0
       return [ distances , Cluster_Metric ] 



def display_cluster(X,km=[],num_clusters=0):
    color = 'brgcmyk'  #List colors
    alpha = 0.5  #color obaque
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1],c = color[0],alpha = alpha,s = s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[km.labels_==i,0],X[km.labels_==i,1],c = color[i],alpha = alpha,s=s)
            plt.scatter(km.cluster_centers_[i][0],km.cluster_centers_[i][1],c = color[i], marker = 'x', s = 100)


# prepare the figure sise and background 
# this part can be replaced by a number of subplots 
plt.rcParams['figure.figsize'] = [8,8]
sns.set_style("whitegrid")
sns.set_context("talk")
# Produce a data set that represent the x and y o coordinates of a circle 
# this part can be replaced by data that you import froma file 
angle = np.linspace(0,2*np.pi,20, endpoint = False)
X = np.append([np.cos(angle)],[np.sin(angle)],0).transpose()
# Data is displayed 
# to display the data only it is assumed that the number of clusters is zero which is the default of the fuction 
display_cluster(X)



# n_samples = 1000
# n_bins = 4  
# centers = [(-3, -3), (0, 0), (3, 3), (6, 6), (9,9)]
# X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
#                   centers=centers, shuffle=False, random_state=42)
# display_cluster(X)



n_samples = 1000
X, y = noisy_moons = make_moons(n_samples=n_samples, noise= .1)
display_cluster(X)