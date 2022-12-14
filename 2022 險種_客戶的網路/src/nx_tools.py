import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns

def edges2df(G):
    edges = G.edges()
    projected_edges = pd.DataFrame([{"src": s, "dst": d, **v} for s, d, v in edges.data()])
    return projected_edges

def centrality(G, distance = None):
	return pd.DataFrame([
		nx.degree_centrality(G),
		nx.betweenness_centrality(G, weight = distance),
		nx.closeness_centrality(G, distance = distance),
		], index = ['degree', 'betweeness', 'closeness']).T


def plotly_network(edge_labels, pos, edge_shown_thresh = 100, show_edge_text = True, filename = 'test.html'):
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
	nodes = go.Scatter(x = pos_[:, 0], 
						 y = pos_[:, 1],
						 mode = 'markers+text',
						 marker = {'color': '#020202'},
						 textposition = 'top center',
						 text = node_labels,
						 hovertemplate = '%{text}')
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
							 f'{w:,.0f}',
							 f'{e[0]}-{e[1]}'],)
	# 畫連結hover資訊
	edge_hovers = go.Scatter(x = [i[0] for i in edges_hovers],
							 y = [i[1] for i in edges_hovers],
							 text = [i[2] for i in edges_hovers],
							 meta = [i[3] for i in edges_hovers],
							 mode = 'markers+text' if show_edge_text else 'markers', 
							 textposition = 'middle center',
							 textfont = {'color': '#100F89'},
							 marker = {'opacity': 0},
							 hovertemplate = '%{meta}: %{text}',
							 showlegend = False)
	fig = go.Figure([*edges, nodes, edge_hovers])
	pyo.plot(fig, filename = filename,auto_open = False)


def neightboring_edges(G, n, diff_cate = True, code_cate = None):
	"""
	跟點n相連的edges，適用於無向圖
	diff_cate: 判斷是否只回傳不同險別的edges
	"""
	hit_edges = []
	for u, v, a in G.edges(data = True):
		mask_connect = n in (u, v)
		mask_diffCate = True
		if diff_cate:
			if code_cate is None:
				raise ValueError('If diff_cate is True, argument code_cate is necessary.')
			opp = u if n == v else v
			n_type = code_cate.get(n, {}).get('cate')
			opp_type = code_cate.get(opp, {}).get('cate')
			if n_type == opp_type:
				mask_diffCate = False
		if mask_connect and mask_diffCate:
			hit_edges.append((u, v, a))
	return hit_edges