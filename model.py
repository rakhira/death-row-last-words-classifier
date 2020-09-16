import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('data/CPS_df.csv')

X = df.iloc[:,5:]
y = df['Percent of Kids in Consummated Adoptions']