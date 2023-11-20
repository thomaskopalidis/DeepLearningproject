import networkx as nx
import matplotlib.pyplot as plt
G = nx.karate_club_graph()

# G is an undirected graph
type(G)
print(type(G))


# Visualize the graph
g= nx.draw(G, with_labels = True)
plt.show() # display the graph in a pop-up window




#QUESTION 1: WHAT IS THE AVERAGE DEGREE OF THE KARATE CLUB NETWORK?
def average_degree(num_edges, num_nodes):
  #Calculate the average degree using the formula
  avg_degree = (2* num_edges) / num_nodes
  #round the result to the nearest ingteger
  print(num_edges)
  print(num_nodes)
  print(avg_degree)
  avg_degree = round(avg_degree)

  return avg_degree

num_edges = G.number_of_edges()
num_nodes = G.number_of_nodes()
avg_degree = average_degree(num_edges, num_nodes)
print("Average degree of karate club network is {}".format(avg_degree))



#Q2. WHAT IS THE AVERAGE CLUSTERIN COEFFICIENT OF THE KCN ?
def average_clustering_coefficient(G):
  avg_cluster_coef = nx.average_clustering(G)
  #Round the result to 2 decimal places
  average_clustering_coefficient = round(avg_cluster_coef,2)
  print(avg_cluster_coef)
  return average_clustering_coefficient
avg_cluster_coef = average_clustering_coefficient(G)
print("Average clustering coefficient of karate club network is {}".format(avg_cluster_coef))

#Q3. WHAT IS THE PAGERANK VALUE FOR NODE 0 AFTER ONE PAGERANK INTERATION?
def one_iter_pagerank(G, beta, r0, node_id):
    # TODO: Implement this function that takes a nx.Graph, beta, r0 and node id.
    # The return value r1 is one interation PageRank value for the input node.
    # Please round r1 to 2 decimal places.

    r1 = 0

    # Iterate through the neighbors of node_id
    for neighbor in G.neighbors(node_id):
        degree = G.degree(neighbor)  # Use degree for undirected graphs
        contribution = beta * r0 / degree
        r1 += contribution

    #Add the random jump component
    N = G.number_of_nodes()
    random_jump = (1- beta) / N
    r1 += random_jump

    # Round the result to 2 decimal places
    r1 = round(r1,2)
    return r1

beta = 0.8
r0 = 1 / G.number_of_nodes()
node = 0
r1 = one_iter_pagerank(G, beta, r0, node)
print("The PageRank value for node 0 after one iteration is {}".format(r1))