import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)

bio = pd.read_csv('data/bioclim_data.csv')
print('bioclim cols:', list(bio.columns))
e = bio[bio.Region=='East'].sort_values('Year')
print('\n=== East: BIO01 (annual mean temp K), BIO12 (precip), 2010-2025 ===')
print(e[e.Year>=2008][['Year','BIO01_hadgem2','BIO05_hadgem2','BIO12_hadgem2','BIO16_hadgem2','cloud-cover_annual-mean_hadgem2']].round(4).to_string(index=False))

# Is bioclim just a smooth trend? autocorrelation / relation to real known years
print('\n=== year-to-year corr of BIO01 across regions (is it a shared GCM realization?) ===')
p = bio.pivot_table(index='Year', columns='Region', values='BIO01_hadgem2')
print(p.corr().round(2).to_string())

print('\n=== Known real-weather check: 2012 was famously wet in UK, 2022 famously hot/dry ===')
uk = bio.groupby('Year')[['BIO01_hadgem2','BIO12_hadgem2','BIO05_hadgem2']].mean()
print(uk.loc[2008:2025].round(4).to_string())

# variance decomposition: how much of bioclim variance is trend vs interannual
print('\n=== detrended interannual sd vs trend, East ===')
for c in ['BIO01_hadgem2','BIO12_hadgem2','BIO04_hadgem2']:
    y = e.set_index('Year')[c].dropna()
    tr = np.polyval(np.polyfit(y.index, y.values, 1), y.index)
    print(f'{c:22s} sd={y.std():.4g} resid_sd={np.std(y.values-tr):.4g} trend_frac={1-np.var(y.values-tr)/np.var(y.values):.2f}')
