import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from search_utils import build_graph, gbfs, a_star  # your reusable functions

st.title("Informed Search Visualizer: GBFS vs A*")

G, coordinates = build_graph()  # same graph used in Tasks 1-3

nodes = list(G.nodes())
start = st.selectbox("Start node", nodes, index=0)
goal = st.selectbox("Goal node", nodes, index=len(nodes) - 1)
algorithm = st.selectbox("Algorithm", ["GBFS", "A*"])

if st.button("Run Search"):
    if algorithm == "GBFS":
        expansion_order, path, cost = gbfs(G, coordinates, start, goal)
    else:
        expansion_order, path, cost, _ = a_star(G, coordinates, start, goal)

    pos = nx.get_node_attributes(G, "pos")
    path_edges = list(zip(path, path[1:]))

    fig, ax = plt.subplots(figsize=(8, 6))
    node_colors = ["#ff6b6b" if n == start else "#ffd93d" if n == goal else "#8ecae6" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1400, edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color="gray", width=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=3.5, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "weight"), font_size=8, ax=ax)
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("Results")
    st.write(f"**Algorithm:** {algorithm}")
    st.write(f"**Solution path:** {' → '.join(path)}")
    st.write(f"**Total path cost:** {cost}")
    st.write(f"**Nodes expanded (in order):** {expansion_order}")