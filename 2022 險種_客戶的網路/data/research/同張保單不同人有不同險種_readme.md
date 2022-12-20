同張保單不同人有不同險種.csv

## Q: 同張保單、不同的被保人，是否有多種iins_kind?
## A: 有，同一張保單險種細項會有所不同。 但大險種都相同

```python
a = df_ply.groupby(['ipolicy', 'assured_id'])['iins_kind'].agg(['unique', 'nunique'])
df_ply['iins_kind_ply'] = df_ply['iins_kind'].str.slice(0, 2)
a = df_ply.groupby(['ipolicy', 'assured_id'])['iins_kind'].agg(['unique', 'nunique'])
a['unique'] = a['unique'].apply(lambda x: ','.join(x))
b = a.groupby(['ipolicy'])['unique'].nunique()
a.query("ipolicy.isin(@b[@b>1].index)").to_csv(os.path.join(PROJECT_PATH, 'data/research/同張保單不同人有不同險種.csv'), sep = '|')
```