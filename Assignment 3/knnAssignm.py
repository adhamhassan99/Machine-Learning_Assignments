import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
# %matplotlib inline

df=testdf = pd.read_csv("C:\\Users\Adham\\Documents\\Machine Learning\\Machine-Learning_Assignments\\Assignment 3\\Classified Data")


scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))