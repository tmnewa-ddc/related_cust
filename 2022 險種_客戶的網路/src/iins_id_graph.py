# %%
import pandas as pd
import nx_tools as nt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import glob, re, json, os

PROJECT_PATH = r'D:\新安\2022 網路關聯\2022 險種_客戶的網路'
PATH_PLY_MAIN = r'D:\新安\data\ml_2021\ml_ply_mix.txt'

DIR_HTH = r'D:\新安\data\HTH\hth_policy'
DIR_CAR = r'D:\新安\data\CAR'

path_hth = glob.glob(DIR_HTH + '\*\ht_ply_member_ins.txt')
path_car_ply = glob.glob(DIR_CAR + '\*\*\policy.txt')
path_car_iins = glob.glob(DIR_CAR + '\*\*\ply_ins_type.txt')

PATH_HTH_CODE = r'D:\新安\data\代碼\HTH_險種代碼.csv'
PATH_CAR_CODE_LV1 = r'D:\新安\data\代碼\car_code_iinsType.json'
PATH_CAR_CODE_LV2 = r'D:\新安\data\代碼\car_code_iinsType_detail.json'

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
	ply = load_data(paths_ply, year, usecols = ['ipolicy', 'fassured', 'iassured'], **kwargs)
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

def load_hth_data(paths_hth, year, **kwargs):
	hth = load_data(paths_hth, year, usecols = ['ipolicy', 'assured_id', 'iins_kind'], **kwargs)
	hth['iins_type'] = hth['iins_kind'].str.slice(0, 2)
	hth = hth.merge(pd.read_csv(PATH_HTH_CODE).dropna().rename(columns = {'名稱': 'iins_type_name', '險別': 'iins_type_cate'}),
			  		left_on = 'iins_type', right_on = '代碼', how = 'left')\
			 .rename(columns = {'assured_id': 'iassured'})\
			 .assign(iins_cate = 'HTH')
	return hth

# %%
# load & prepare data
car = load_car_data(path_ply, path_iins, 2020)
hth = load_hth_data(path_hth, 2020)
data = pd.concat([car, hth])

hth_code = pd.read_csv(PATH_HTH_CODE).dropna().rename(columns = {'名稱': 'iins_type_name', '險別': 'iins_type_cate'})
hth_code = hth_code.set_index('代碼')['iins_type_name'].to_dict()
with open(PATH_CAR_CODE_LV2, 'r', encoding = 'utf-8') as f:
	car_code = json.load(f)
code_mapper = {**hth_code, **car_code}

# build a graph
iins_types = data['iins_type'].unique()
G = nx.from_pandas_edgelist(data, source = 'iassured', target = 'iins_type')
# for cate of nodes
code_cate = data.set_index('iins_type')['iins_cate'].to_dict()
code_cate = {k: {'cate': v} for k, v in code_cate.items()}
nx.set_node_attributes(G, code_cate)
# ---------------------------------------------------------------- #
# 險別之間的連結: 共有的被保人有多少位?
# ---------------------------------------------------------------- #
projected_G = nx.algorithms.bipartite.projected_graph(G, nodes = iins_types)
w_projected_G = nx.algorithms.bipartite.weighted_projected_graph(G, nodes = iins_types)
# 距離權重 = 權重的倒數
updated_weights = {(e[0], e[1]): {**e[2], **{'dist': 1/e[2]['weight']}}
					for e in w_projected_G.edges(data = True)}
nx.set_edge_attributes(w_projected_G, updated_weights)
# graph visialization : projected graph (iins_type)
edge_weights = nx.get_edge_attributes(w_projected_G, 'weight') 
pos = graphviz_layout(projected_G, prog=r'D:\Graphviz\bin\twopi.exe')
nt.plotly_network(edge_weights, pos = pos, show_edge_text = False, edge_shown_thresh = 1000)
# dataframe projection (iins_type)
df_projected = nt.edges2df(w_projected_G)
df_projected = df_projected.assign(src_name = lambda x: x['src'].apply(lambda y: code_mapper.get(y, y)),
							       dst_name = lambda x: x['dst'].apply(lambda y: code_mapper.get(y, y)))
# 跨險別
mask_crossing_1 = df_projected['src'].isin(hth_code.keys()) & df_projected['dst'].isin(car_code.keys())
mask_crossing_2 = df_projected['dst'].isin(hth_code.keys()) & df_projected['src'].isin(car_code.keys())
mask_crossing = mask_crossing_1 | mask_crossing_2
df_projected[mask_crossing].sort_values('weight', ascending = False)
# %%

# ---------------------------------------------------------------- #
# 險別之間的連結: 每個險別與其他險別的連接程度(中心性)?
# degree: 連接的其他險別 / (最大的可能連結束, i.e. 所有險別數-1)
# betweeness: 通過此點u的最短路徑數 / 所有最短路徑數 = 越大的表示此點u對連結越重要
# closeness: 1/ avg(抵達點u所有的其他點v_s的平均最短路徑) = 越大表示離越容易抵達
#
# .......................
#
# 除了整體，我還想知道連結其他險別的情況 (i.e. 除了點u之外，先排除所有與點u同險別的節點
#
# ---------------------------------------------------------------- #
centrality = nt.centrality(w_projected_G, distance = 'dist')
neighbors_custNodes = []
for i in centrality.index:
	nn = [n for n in G.neighbors(i) if n not in iins_types]
	neighbors_custNodes.append(nn)	
centrality['customers'] = neighbors_custNodes
centrality['n_custtomers'] = centrality['customers'].apply(len)
centrality.sort_values('betweeness', ascending = False)
centrality['name'] = [code_mapper.get(i, i) for i in centrality.index]
# 連結的其他險別中，前三名重要的
holder = []
for n in centrality.index:
	vals = {'top1': None, 'top1_name': None, 'top1_weight': None, 
		    'top2': None, 'top2_name': None,  'top2_weight': None, 
		    'top3': None, 'top3_name': None,  'top3_weight': None,}
	neighboring_edges = nt.neightboring_edges(w_projected_G, n, diff_cate = True, code_cate = code_cate)
	tops = sorted(neighboring_edges, key = lambda x: x[2].get('weight'), reverse = True)[:3]
	for i, t in enumerate(tops, start = 1):
		opp = t[0] if n != t[0] else t[1]
		vals[f'top{i}'] = opp
		vals[f'top{i}_name'] = code_mapper.get(opp)
		vals[f'top{i}_weight'] = t[2].get('weight')
	holder.append(vals)
centrality = pd.concat([centrality, pd.DataFrame(holder, index = centrality.index)], axis = 1)

centrality.drop('customers', axis = 1)\
		  .sort_values('top1_weight', ascending = False)\
		  .reset_index()\
		  .to_csv(os.path.join(PROJECT_PATH, 'data/research/險種投影中心性.csv'))
		  
centrality