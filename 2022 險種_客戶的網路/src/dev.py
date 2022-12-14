# %%
import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import os, glob, re
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns


PROJECT_PATH = r'D:\新安\2022 險種_客戶的網路'

DIR_PLY = r'D:\新安\data\HTH\hth_policy'
PATH_HTH_CODE = r'D:\新安\data\代碼\HTH_險種代碼.csv'


PATH_PLY_CHRIS = r'D:\新安\data\ml_2021\ml_ply_mix.txt'

# %%
# load data path
ply_paths = glob.glob(DIR_PLY + r'\*\ht_ply_member_ins.txt')
claim_path = os.path.join(DIR_CLAIM, 'ht_appraisal.txt')
df_claim = pd.read_csv(claim_path, sep = '|', dtype = str)
hth_code = pd.read_csv(PATH_HTH_CODE).dropna().set_index('代碼')['名稱'].to_dict()
df_ml = pd.read_csv(PATH_PLY_CHRIS, sep = '|', dtype = str)#, usecols = ['ipolicy', 'id', 'cate', 'iroute_full'])
df_ml.columns
# load data
def load_data(year, df_claim):

	df_ply = pd.read_csv([p for p in ply_paths if re.match(f'.*{year}.*', p)][0],
						 sep = '|',
						 dtype = 'str')
	df_m = df_ply.merge(df_claim[['ipolicy', 'iassured', 'iclaim']], 
						left_on = ['ipolicy', 'assured_id'],
						right_on = ['ipolicy', 'iassured'],
				 		how = 'left')\
				 .merge(df_ml, on = ['ipolicy'], how = 'left')
	df_m['iins_kind_ply'] = df_m['iins_kind'].str.slice(0, 2)
	return df_m

def edges2df(G):
    edges = G.edges()
    projected_edges = pd.DataFrame([{"src": s, "dst": d, **v} for s, d, v in edges.data()])
    return projected_edges

def centrality(G):
	return pd.DataFrame([
		nx.degree_centrality(G),
		nx.betweenness_centrality(G),
		nx.closeness_centrality(G),
		], index = ['degree', 'betweeness', 'closeness']).T

# %%	
data = load_data('2020', df_claim)

sub_data = data.query("cate not in ['H_TV', 'H_CV']")
iins_kind = sub_data['iins_kind_ply'].unique()
G = nx.from_pandas_edgelist(sub_data, source = 'assured_id', target = 'iins_kind_ply')

# %%
projected_G = nx.algorithms.bipartite.projected_graph(G, nodes = iins_kind)
# 兩個險種共有的被保人人數
w_projected_G = nx.algorithms.bipartite.weighted_projected_graph(G, nodes = iins_kind)
# %% 
# verify w_projected_G: # 兩個險種共有的被保人人數為什麼這麼少!?
edge_weights = nx.get_edge_attributes(w_projected_G,'weight') 
{k: v for k, v in sorted(edge_weights.items(), key = lambda x: x[1], reverse = True)}

holder = {}
for src, tar in edge_weights.keys():
	id_src = sub_data.query('iins_kind_ply == @src')['assured_id'].unique()
	id_tar = sub_data.query('iins_kind_ply == @tar')['assured_id'].unique()
	overlap = set(id_src).intersection(id_tar)
	holder[(src, tar)] = len(overlap)

{k: v for k, v in edge_weights.items() if v != holder.get(k)}

# %%
# network plotly

edge_weights = nx.get_edge_attributes(w_projected_G,'weight')
pos = graphviz_layout(projected_G, prog=r'D:\Graphviz\bin\twopi.exe')
edge_labels = edge_weights

def plotly_network(edge_labels, pos, edge_shown_thresh = 100):
	def __get_color(edge_labels):
		palette = pd.DataFrame(edge_labels.values(), index = edge_labels.keys(), columns = ['edge'])
		uq_vals = sorted(palette['edge'].unique())
		cmap = {k: f'rgb({v[0]},{v[1]},{v[2]})' 
					for k, v in zip(uq_vals, sns.color_palette('Reds', len(uq_vals)))}
		palette['color'] = [cmap.get(e) for e in palette['edge']]
		return palette['color']
	# 畫nodes
	pos_ = np.array((list(pos.values())))
	node_labels = list(pos.keys())
	node_nominal = [hth_code.get(n, n) for n in node_labels]
	nodes = go.Scatter(x = pos_[:, 0], 
						 y = pos_[:, 1],
						 mode = 'markers+text',
						 marker = {'color': '#020202'},
						 textposition = 'top center',
						 text = node_labels,
						 meta = node_nominal,
						 hovertemplate = '%{text}: %{meta}')
	# 畫連結
	edges = [] # 所有連結
	edges_hovers = [] # 連結的hover資訊(在連結中間加個看不見的點)
	edge_labels_2plot = {k: v for k, v in edge_labels.items() if v >= edge_shown_thresh}
	colors = __get_color(edge_labels_2plot)
	for c, (e, w) in zip(colors, edge_labels_2plot.items()):
		src = pos.get(e[0], [])
		tar = pos.get(e[1], [])
		edges.append(go.Scatter(x = [src[0], tar[0]], 
				   				y = [src[1], tar[1]],
				   				mode = 'lines',
							    line = {'color': c, 'width': max(np.log(w)/2, 3)},
							    hovertemplate = f'{e[0]}-{e[1]}: w',
							    showlegend = False)
								)

		edges_hovers.append([(src[0]+tar[0])/2, 
							 (src[1]+tar[1])/2,
							 f'{w:,.0f}'])
	# 畫連結hover資訊
	edge_hovers = go.Scatter(x = [i[0] for i in edges_hovers],
							 y = [i[1] for i in edges_hovers],
							 text = [i[2] for i in edges_hovers],
							 mode = 'markers+text',
							 textposition = 'middle center',
							 textfont = {'color': '#100F89'},
							 marker = {'opacity': 0},
							 showlegend = False)
	fig = go.Figure([*edges, nodes, edge_hovers])
	pyo.plot(fig, filename = os.path.join(PROJECT_PATH, 'test.html'),auto_open = False)

# %%


G, pos
centrality(projected_G).sort_values('degree', ascending = False)
df_projected = edges2df(projected_G)
df_projected

data['iins_kind_ply'].nunique()

