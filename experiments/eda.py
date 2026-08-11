import pandas as pd, numpy as np
pd.set_option('display.width', 200)

pest = pd.read_csv('data/pest_data.csv')
agro = pd.read_csv('data/agronomic_data.csv')
bio  = pd.read_csv('data/bioclim_data.csv')
fung = pd.read_csv('data/fungicide_data.csv')
luc  = pd.read_csv('data/prop_LUC.csv')

for name, df in [('pest',pest),('agro',agro),('bio',bio),('fung',fung),('luc',luc)]:
    print(f'=== {name}: shape={df.shape}')
    print('  years:', df.Year.min(), '->', df.Year.max(), '| n_years:', df.Year.nunique())
    print('  regions:', sorted(df.Region.unique()))
    print('  n per year (last 10):')
    print(df.groupby('Year').size().tail(10).to_dict())
    print()
