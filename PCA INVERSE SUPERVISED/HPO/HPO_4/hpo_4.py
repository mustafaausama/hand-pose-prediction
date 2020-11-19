import pyrolite
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df = pd.read_csv("hpo_4.csv")
df.columns = ['num_unit1', 'num_unit2','lr', 'loss']
print(df)

#parallel_coordinates(data, class_column='loss')
pyrolite.plot.parallel.parallel(df)
plt.figure(figsize=(100,100))
plt.savefig('fig4.png', dpi = 100)
plt.show()







