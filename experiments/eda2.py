import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)

pest = pd.read_csv('data/pest_data.csv')
TARGETS = [c for c in pest.columns if c.startswith(('L1_','L2_'))]

print('2026 row sample:'); print(pest[pest.Year==2026].head(3).to_string()); print()
pest = pest.replace(-9999, np.nan)
print('=== missing years in pest (per region) ===')
allyr = set(range(1971, 2027))
for r, g in pest.groupby('Region'):
    miss = sorted(allyr - set(g.Year))
    print(f'{r:28s} n={len(g):3d} yrs {g.Year.min()}-{g.Year.max()} missing: {miss}')
print()
print('=== target stats (all years) ===')
print(pest[TARGETS].describe().T[['count','mean','std','min','50%','max']].round(3))
print()
print('=== target stats (2015+) ===')
print(pest[pest.Year>=2015][TARGETS].describe().T[['count','mean','std','min','50%','max']].round(3))
print()
print('=== per-year means, 2010+ ===')
print(pest[pest.Year>=2010].groupby('Year')[TARGETS].mean().round(2).to_string())
