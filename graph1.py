import networkx as nx
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
print("node degree")
for v in G:
    print(f"{v:4} {G.degree(v):6}")
nx.draw_circular(G, with_labels = True)
plt.show()