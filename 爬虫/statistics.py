import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('TV-data.csv')
df=df.drop(df[(df.up_speed==0) | (df.down_speed==0)].index)
print(df['down_speed'].mean(axis=0)/1024,df['down_speed'].max(axis=0)/1024,df['down_speed'].min(axis=0)/1024,df['down_speed'].std(axis=0)/1024)
df.head(60).plot()
plt.show()
