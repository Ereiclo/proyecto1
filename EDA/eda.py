import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename_train = '../proyect_dataset/training.csv'
filename_test = '../proyect_dataset/testing.csv'

df_train = pd.read_csv(filename_train)
df_test = pd.read_csv(filename_test)

print(df_train.head())
print(df_train.info())

print(df_train.describe())

sns.histplot(df_train['CICLOS'])
plt.show()

sns.pairplot(df_train)
plt.show()

corr = df_train.corr()
sns.heatmap(corr, annot=True)
plt.show()


import klib
import pandas as pd

df = pd.read_csv(filename_train)
klib.missingval_plot(df)

df_cleaned_train = klib.data_cleaning(df)

# matriz de correlación
klib.corr_plot(df_cleaned_train, annot=False, figsize=(10,10))

klib.corr_plot(df_cleaned_train, split='pos', annot=False)

klib.corr_plot(df_cleaned_train, target='clase')

# testing
df = pd.read_csv(filename_test)
df_cleaned_test = klib.data_cleaning(filename_test)

klib.corr_plot(df_cleaned_test, annot=False, figsize=(10,10))

klib.corr_plot(df_cleaned_test, split='pos', annot=False)

klib.corr_plot(df_cleaned_test, target='clase')