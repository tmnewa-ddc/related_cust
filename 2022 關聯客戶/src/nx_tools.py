# %%
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


def plotly_network(edge_labels, 
				   pos, 
				   pos_colors = None,
				   pos_symbols = None, 
				   pos_edges = None,
				   edge_shown_thresh = 100., 
				   show_edge_text = True, 
				   filename = 'test.html', 
				   return_fig = False, 
				   marker_size = 10):
	"""
	edge_labels: 連結的權重; dict; {(src, dst): weight}; 可以用nx.get_edge_attributes(G, 'weight')取得
	pos: nodes的位置; dict; {node_name: (x, y)}
	pos_colors: nodes的顏色; dict; {node_name: ColorCode}
	edge_shown_thresh: 連結的權重多少以上才顯示(為求畫面整潔); float
	show_edge_text: 要不要顯示連結權重

	note: 小心node_names的型別必須和src, dst一致
	"""
	def __get_edgeCmap(edge_labels):
		palette = pd.DataFrame(edge_labels.values(), index = edge_labels.keys(), columns = ['edge'])
		uq_vals = sorted(palette['edge'].unique())
		cmap = {k: f'rgb({v[0]},{v[1]},{v[2]})' 
					for k, v in zip(uq_vals, sns.color_palette('Reds', len(uq_vals)))}
		palette['color'] = [cmap.get(e) for e in palette['edge']]
		return palette['color']
	# 畫nodes
	pos_ = np.array((list(pos.values())))
	node_labels = list(pos.keys())
	colors = '#020202' if pos_colors is None else [pos_colors.get(n, '#020202') for n in pos.keys()]
	symbols = 'circle' if pos_symbols is None else [pos_symbols.get(n, 'circle') for n in pos.keys()]
	mk_edge = None if pos_edges is None else [pos_edges.get(n) for n in pos.keys()]
	nodes = go.Scatter(x = pos_[:, 0], 
						 y = pos_[:, 1],
						 mode = 'markers+text',
						 marker = {'color': colors, 'size': marker_size, 'symbol': symbols, 'line': mk_edge},
						 textposition = 'top center',
						 text = node_labels,
						 hovertemplate = '%{text}')
	# 畫連結
	edges = [] # 所有連結
	edges_hovers = [] # 連結的hover資訊(在連結中間加個看不見的點)
	edge_labels_2plot = {k: v for k, v in edge_labels.items() if v >= edge_shown_thresh}
	colors = __get_edgeCmap(edge_labels_2plot)
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
	if return_fig:
		return fig
	else:
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


# %%

if __name__ == '__name__':
	from networkx.drawing.nx_pydot import graphviz_layout
	from plotly.subplots import make_subplots

	G = nx.Graph()
	G.add_edges_from([(1, 2), (1, 3), (3, 4), (3, 5)])
	pos_colors = {1: '#EB3434', 2: '#3574BB', 3: '#3574BB'}
	pos_edges = {1: {'width': 5}, 3: {'width': 5}}

	pos = graphviz_layout(G, prog=r'D:\Graphviz\bin\dot.exe')
	pos = {int(n): p for n, p in pos.items()}
	el = {e: 1 for e in G.edges()}
	fig_ori = plotly_network(el, pos, pos_colors, pos_edges = pos_edges , edge_shown_thresh = 0, show_edge_text = False, return_fig = True, marker_size = 20)

	G_projected = nx.algorithms.bipartite.projected_graph(G, nodes = [1, 2, 3 ,4 ,5])
	pos = graphviz_layout(G_projected, prog=r'D:\Graphviz\bin\dot.exe')
	pos = {int(n): p for n, p in pos.items()}
	el = {e: 1 for e in G_projected.edges()}
	fig_projected = plotly_network(el, pos, pos_colors, pos_edges = pos_edges ,edge_shown_thresh = 0, show_edge_text = False, return_fig = True, marker_size = 20)

	fig = make_subplots(1, 2)
	for t in fig_ori.data:
		fig.add_trace(t, row = 1, col = 1)
	for t in fig_projected.data:
		fig.add_trace(t, row = 1, col = 2)
	pyo.plot(fig, filename = 'test.html', auto_open = False)


# %%