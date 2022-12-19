# %%
import pandas as pd
import numpy as np
import nx_tools as nt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import glob, re, json, os

PROJECT_PATH = r'D:\新安\2022 網路關聯'
PATH_PLY_MAIN = r'D:\新安\data\ml_2021\ml_ply_mix.txt'
PATH_PLY_MAIN_ALL = r'D:\新安\data\訂單整合資料\car_policy_full.txt'

DIR_HTH = r'D:\新安\data\HTH\hth_policy'
DIR_CAR = r'D:\新安\data\CAR'

path_hth = glob.glob(DIR_HTH + '\*\ht_ply_member_ins.txt')
path_hth_ply = glob.glob(DIR_HTH + '\*\ht_policy.txt')
path_car_ply = glob.glob(DIR_CAR + '\*\*\policy.txt')
path_car_iins = glob.glob(DIR_CAR + '\*\*\ply_ins_type.txt')

PATH_HTH_CODE = r'D:\新安\data\代碼\HTH_險種代碼.csv'
PATH_CAR_CODE_LV1 = r'D:\新安\data\代碼\car_code_iinsType.json'
PATH_CAR_CODE_LV2 = r'D:\新安\data\代碼\car_code_iinsType_detail.json'

PATH_PROFILE = r'D:\新安\2022 客戶分群\data\features.parq'
DIR_NODES_SAVED = r'D:\新安\2022 網路關聯\2022 關聯客戶\data'

# %%

path_ply, path_iins = path_car_ply, path_car_iins
paths = path_car_ply
year = 2021
def load_data(paths, year, usecols = [], sep = '|'):
	filter_paths = [p for p in paths if re.match(f'.*{year}.*', p)]
	data = []
	for p in filter_paths:
		if usecols:
			data.append(pd.read_csv(p, sep = sep, dtype = str, usecols = usecols))
		else:
			data.append(pd.read_csv(p, sep = sep, dtype = str))
	return pd.concat(data)

def load_car_data(paths_ply, paths_iins, year, **kwargs):
	ply = load_data(paths_ply, year, usecols = ['ipolicy', 'fassured', 'iassured', 'iapplicant', 'itag'], **kwargs)
	iins = load_data(path_car_iins, year, usecols = ['ipolicy', 'iins_type', 'mins_premium'], **kwargs)
	m = iins.merge(ply, on = 'ipolicy', how = 'left')
	m = m.assign(mins_premium = lambda x: pd.to_numeric(x['mins_premium'], errors = 'ignore'))
	m = m.query("fassured == '1' & mins_premium > 0")
	with open(PATH_CAR_CODE_LV1, 'r', encoding = 'utf-8') as f:
		iins_mapper_lv1 = json.load(f)
	with open(PATH_CAR_CODE_LV2, 'r', encoding = 'utf-8') as f:
		iins_mapper_lv2 = json.load(f)
	m = m.assign(iins_type_name = lambda x: x['iins_type'].apply(lambda i: iins_mapper_lv1.get(i)),
				 iins_type_name_detail = lambda x: x['iins_type'].apply(lambda i: iins_mapper_lv2.get(i)),
				 iins_cate = 'CAR') 
	return m

def load_hth_data(paths_hth, paths_hth_ply, year, **kwargs):
	hth = load_data(paths_hth, year, usecols = ['ipolicy', 'assured_id', 'iins_kind'], **kwargs)
	hth_ply = load_data(path_hth_ply, year, usecols = ['ipolicy', 'applicant_id'])
	hth['iins_type'] = hth['iins_kind'].str.slice(0, 2)
	hth = hth.merge(hth_ply, on = 'ipolicy', how = 'left')\
			 .merge(pd.read_csv(PATH_HTH_CODE).dropna().rename(columns = {'名稱': 'iins_type_name', '險別': 'iins_type_cate'}),
			  		left_on = 'iins_type', right_on = '代碼', how = 'left')\
			 .rename(columns = {'assured_id': 'iassured', 'applicant_id': 'iapplicant'})\
			 .assign(iins_cate = 'HTH')
	return hth


# ------------------------------------------------------- #
# 各年度的關聯客戶 (因為資料太大，拆年度執行)
# ------------------------------------------------------- #
# %%
# load & prepare data
for year in range(2017, 2022):
	car = load_car_data(path_ply, path_iins, year)
	hth = load_hth_data(path_hth, path_hth_ply, year)
	# ======================================================= #
	# 小心了，最後做出來的關聯被保人會不知道是哪個險別來的
	# ======================================================= #
	data = pd.concat([car, hth])

	hth_code = pd.read_csv(PATH_HTH_CODE).dropna().rename(columns = {'名稱': 'iins_type_name', '險別': 'iins_type_cate'})
	hth_code = hth_code.set_index('代碼')['iins_type_name'].to_dict()
	with open(PATH_CAR_CODE_LV2, 'r', encoding = 'utf-8') as f:
		car_code = json.load(f)
	code_mapper = {**hth_code, **car_code}

	# build a graph
	data = data.query("iins_type_cate != '團傷'")
	iapplicant = data['iapplicant'].unique()
	iassured = data['iassured'].unique()

	# %%
	# ---------------------------------------------------------------- #
	# 重要保戶:
	# 投影的鄰居: 被保人-->|要保人|--被保人
	# 有關係的客戶 = 原本的鄰居.union(投影的鄰居)
	# ---------------------------------------------------------------- #
	G = nx.from_pandas_edgelist(data, source = 'iapplicant', target = 'iassured')#, create_using = nx.DiGraph)
	projected_G = nx.algorithms.bipartite.projected_graph(G, nodes = iassured)

	used_itag_ias = data.groupby('itag')['iassured'].nunique()
	used_itag_iap = data.groupby('itag')['iapplicant'].nunique()
	used_itag = set(used_itag_ias.index).union(used_itag_iap.index)
	data_usedTag = data.query("itag.isin(@used_itag)")
	iassured_tag = data_usedTag['iassured'].unique()
	G_tag = nx.from_pandas_edgelist(data_usedTag, source = 'itag', target = 'iassured')
	projected_GTag = nx.algorithms.bipartite.projected_graph(G_tag, nodes = iassured_tag)
	# %%
	nodes = pd.DataFrame([[set([j for j in G.neighbors(i) if j != i and isinstance(j, str)]) for i in iassured],
						  [set([j for j in projected_G.neighbors(i) if j != i and isinstance(j, str)]) for i in iassured],
						  [set([j for j in projected_GTag.neighbors(i) if j != i and isinstance(j, str)]) for i in iassured_tag],
						 ], columns = iassured, index = ['直接連接', '通過要保', '通過標的物']).T


	nodes = nodes.assign(num_original = lambda x: x['直接連接'].apply(len),
						 num_byAppl = lambda x: x['通過要保'].apply(len),
						 num_byTag = lambda x: x['通過標的物'].apply(lambda y: len(y) if y else 0))
	nodes['直接連接'] = nodes['直接連接'].apply(lambda x: ','.join(x) if x else '')
	nodes['通過要保'] = nodes['通過要保'].apply(lambda x: ','.join(x) if x else '')
	nodes['通過標的物'] = nodes['通過標的物'].apply(lambda x: ','.join(x) if x else '')


	# %%	
	profile = pd.read_parquet(PATH_PROFILE)
	profile = profile.groupby('id')[['plyAmt', 'clmAmt', 'ipolicy', 'clmed_iply']].sum()
	profile = profile.assign(clm_rate = lambda x: x['clmAmt'] / x['plyAmt'],
							 clm_ratio = lambda x: x['clmed_iply'] / x['ipolicy'])

	nodes = nodes.merge(profile, left_index = True, right_on = 'id', how ='left')

	nodes.sort_values('num_original', ascending = False, inplace = True)

	nodes.sort_values('num_byTag', ascending = False).to_parquet(os.path.join(PROJECT_PATH, f'2022 關聯客戶/data/{year}_關聯客戶.parq'), compression = 'brotli')
	nodes.sort_values('num_byTag', ascending = False).to_csv(os.path.join(PROJECT_PATH, f'2022 關聯客戶/data/{year}_關聯客戶.csv'))

# ------------------------------------------------------- #
# 把每個人多年的資料合成為一筆:
# 1. 關聯客戶 = 多年來累積的不重複關聯各戶 (i.e. 各年的union)
# 2. 保費、賠付、保單等 = 多年來的總和
# ------------------------------------------------------- #
# nodes.reset_index(inplace = True)
path_nodes_years = glob.glob(DIR_NODES_SAVED + '\\*_關聯客戶.parq')
nodes_years = pd.concat([pd.read_parquet(p) for p in path_nodes_years])
nodes_years.reset_index(inplace = True)
nodes_years.columns
def ids_union(ser):
	return set(','.join(ser).split(','))
nodes_years = nodes_years.groupby('index').agg({'直接連接': ids_union,
												'通過要保': ids_union,
												'通過標的物': ids_union,
												'plyAmt': sum,
												'clmAmt': sum,
												'ipolicy': sum,
												'clmed_iply': sum,
												})
nodes_years = nodes_years.assign(num_original = lambda x: x['直接連接'].apply(len),
								 num_byAppl = lambda x: x['通過要保'].apply(len),
								 num_byTag = lambda x: x['通過標的物'].apply(lambda y: len(y) if y else 0),
								 clm_rate = lambda x: x['clmAmt'] / x['plyAmt'],
								 clm_ratio = lambda x: x['clmed_iply'] / x['ipolicy'])

nodes_years.to_parquet(os.path.join(PROJECT_PATH, f'2022 關聯客戶/data/不分年/關聯客戶.parq'), compression = 'brotli')


# ------------------------------------------------------- #
# 為不同方法找到的關聯客戶，計算統計量。用來衡量和指定人相關的一群人是否需要警戒
# e.g. 假設小明直接關聯的有10個人，這10個人的損率等等統計量
# ------------------------------------------------------- #

# index = set(nodes_years.index)
# def row_stats(connected_group: list, cols = ['plyAmt', 'clmAmt', 'ipolicy', 'clmed_iply']):
# 	# res = nodes.query("index.isin(@connected_group)")[['plyAmt', 'clmAmt', 'ipolicy', 'clmed_iply']].sum()
# 	connected_group = index.intersection(connected_group)
# 	res = nodes_years.loc[connected_group][['plyAmt', 'clmAmt', 'ipolicy', 'clmed_iply']].sum()
# 	res['clm_rate'] = res['clmAmt'] / res['plyAmt']
# 	res['clm_ratio'] = res['clmed_iply'] / res['ipolicy']
# 	return res

# gp_stats_1 = nodes_years['直接連接'].apply(row_stats)
# gp_stats_1.to_parquet(os.path.join(PROJECT_PATH, f'2022 關聯客戶/data/不分年/關聯客戶_gp統計_直接連接.parq'), compression = 'brotli')

# gp_stats_1 = nodes_years['通過要保'].apply(row_stats)
# gp_stats_1.to_parquet(os.path.join(PROJECT_PATH, f'2022 關聯客戶/data/不分年/關聯客戶_gp統計_通過要保.parq'), compression = 'brotli')

# gp_stats_1 = nodes_years['通過標的物'].apply(row_stats)
# gp_stats_1.to_parquet(os.path.join(PROJECT_PATH, f'2022 關聯客戶/data/不分年/關聯客戶_gp統計_通過標的物.parq'), compression = 'brotli')

# ------------------------------------------------------- #
# 把不分年的關聯客戶 left join 關聯客戶_gp統計
# ------------------------------------------------------- #
nodes_years = pd.read_parquet(os.path.join(PROJECT_PATH, '2022 關聯客戶/data/不分年/關聯客戶.parq'))
for gp in ['直接連接', '通過標的物', '通過要保']:
	gp_data = pd.read_parquet(os.path.join(PROJECT_PATH, f'2022 關聯客戶/data/不分年/關聯客戶_gp統計_{gp}.parq'))
	gp_data.columns = [f'{c}_{gp}' for c in gp_data.columns]
	nodes_years = nodes_years.merge(gp_data, left_index = True, right_index = True, how = 'left')
nodes_years.iloc[:, 3:].to_parquet(os.path.join(PROJECT_PATH, '2022 關聯客戶/data/不分年/關聯客戶_prepared.parq'), compression = 'brotli')
nodes_years.iloc[:, 3:].to_csv(os.path.join(PROJECT_PATH, '2022 關聯客戶/data/不分年/關聯客戶_prepared.csv'), sep = '|')

# ------------------------------------------------------- #
# 分析
# ------------------------------------------------------- #
nodes_years.iloc[:, 3:].sort_values(['num_original', 'clm_rate'], ascending = False)\
		   .head(10)\
		   .to_markdown(os.path.join(PROJECT_PATH, '2022 關聯客戶/data/tops/直接連結tops.txt'))

nodes_years.iloc[:, 3:].sort_values(['num_byAppl', 'clm_rate'], ascending = False)\
		   .head(10)\
		   .to_markdown(os.path.join(PROJECT_PATH, '2022 關聯客戶/data/tops/透過要保tops.txt'))

nodes_years.iloc[:, 3:].sort_values(['num_byTag', 'clm_rate'], ascending = False)\
		   .head(10)\
		   .to_markdown(os.path.join(PROJECT_PATH, '2022 關聯客戶/data/tops/透過標的物tops.txt'))



row_stats = profile.loc[row['直接連接'].split(',')][['plyAmt', 'clmAmt', 'ipolicy', 'clmed_iply']].sum()\
		.assign(clm_rate = lambda x: x['clmAmt'] / x['plyAmt'],
				clm_ratio = lambda x: x['clmed_iply'] / x['ipolicy'])
# %%

# %%
# 說明用圖
import matplotlib.pyplot as plt

G_t = nx.Graph()
G_t.add_edges_from([('1', '1'), ('1', '3'), ('2', '3')])
pos = graphviz_layout(G_t, prog=r'D:\Graphviz\bin\twopi.exe')
G_t.nodes()

fig, ax = plt.subplots()
nx.draw_networkx_edges(
    G_t,
    pos=pos,
    ax=ax,
    arrows=True,
    arrowstyle="-",
    min_source_margin=5,
    min_target_margin=5,
)









import plotly.graph_objects as go
import plotly.offline as pyo


G_t = nx.Graph()
G_t.add_edges_from([(1, 1), (1, 3), (2, 3)])
type_icon = {'iappl': '🙋<br>要保人', 'iassured': '👱<br>被保人', 'car': '🚗'}
nx.set_node_attributes(G_t, {1: {'type': 'iassured'}, 
							 2: {'type': 'iassured'}, 
							 3: {'type': 'iappl'}
							 })
pos = np.array([(0, 0), (.5, .5), (1, 0)])

nodes = go.Scatter(
			x = pos[:, 0],
			y = pos[:, 1],
			text = [type_icon.get(n[1]['type']) for n in G_t.nodes(data = True)],
			mode = 'text',
			textfont = {'size': 20}
		)

connections = []
for e in G_t.edges():
	e
	t = go.Scatter(
					x = pos[e],
					y = pos[:, 1],
					text = [type_icon.get(n[1]['type']) for n in G_t.nodes(data = True)],
					mode = 'text',
					textfont = {'size': 20}
				)

pyo.plot(go.Figure([nodes]), auto_open = False)


G_t.nodes()[1]

G_t_p = nx.algorithms.bipartite.projected_graph(G_t, nodes = [1, 2, 3])
nx.draw(G_t, with_labels = True)
nx.draw(G_t_p, with_labels = True)




for x in nodes['直接連接']:
	','.join(x) if x else ''