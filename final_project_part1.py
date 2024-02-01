import random
import timeit
import matplotlib.pyplot as plt
import min_heap


class DirectedWeightedGraph:

    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)


def dijkstra(G, source):
    pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {}  # Distance dictionary
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    # Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(source, 0)

    # Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(
                    neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + \
                    G.w(current_node, neighbour)
                pred[neighbour] = current_node
    return dist


def bellman_ford(G, source):
    pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {}  # Distance dictionary
    nodes = list(G.adj.keys())

    # Initialize distances
    for node in nodes:
        dist[node] = float("inf")
    dist[source] = 0

    # Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour):
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
    return dist


def dijkstra_approx(G, source, k):
    pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {}  # Distance dictionary
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    # Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(source, 0)

    relax_counts = {node: 0 for node in G.adj.keys()}

    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        if relax_counts[current_node] >= k:
            continue
        relax_counts[current_node] += 1
        for neighbour in G.adj[current_node]:
            if relax_counts[neighbour] >= k:
                continue
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(
                    neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + \
                    G.w(current_node, neighbour)
                pred[neighbour] = current_node
                relax_counts[neighbour] += 1
    return dist


def bellman_ford_approx(G, source, k):
    pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {}  # Distance dictionary
    nodes = list(G.adj.keys())
    relax = {}

    # Initialize distances
    for node in nodes:
        relax[node] = 0
        dist[node] = float("inf")
    dist[source] = 0

    # Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour) and relax[neighbour] < k:
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
                    relax[neighbour] += 1
    return dist


def total_dist(dist):
    total = 0
    for key in dist.keys():
        total += dist[key]
    return total


def create_random_complete_graph(n, upper):
    G = DirectedWeightedGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j, random.randint(1, upper))
    return G


# Assumes G represents its nodes as integers 0,1,...,(n-1)
def mystery(G):
    n = G.number_of_nodes()
    d = init_d(G)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][j] > d[i][k] + d[k][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


def init_d(G):
    n = G.number_of_nodes()
    d = [[float("inf") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if G.are_connected(i, j):
                d[i][j] = G.w(i, j)
        d[i][i] = 0
    return d


################ Experiment suite 1 ##############

# Fix the value of k of the graph,
# and see how the total distance changes
# as the number of nodes changes for dijkstra and dijkstra_approx.
def exp_a():
    k = 3
    runs = 10
    size = [i for i in range(10, 100)]
    dist_list = []
    dist_approx_list = []
    for n in size:
        dist = 0
        dist_approx = 0
        for _ in range(runs):
            G = create_random_complete_graph(n, 100)
            source = random.randint(0, n - 1)
            dist += total_dist(dijkstra(G, source))
            dist_approx += total_dist(dijkstra_approx(G, source, k))
        dist_list.append(dist / runs)
        dist_approx_list.append(dist_approx / runs)
        print("n: " + str(n) + ", total distance: " + str(dist / runs) +
              ", total distance approx: " + str(dist_approx / runs))
    plt.plot(size, dist_list, label="Dijkstra")
    plt.plot(size, dist_approx_list, label="Dijkstra Approx")
    plt.xlabel("Number of nodes")
    plt.ylabel("Total distance")
    plt.legend()
    plt.show()


# Fix the number of nodes of the graph,
# and see how the total distance changes
# as the value of k changes for dijkstra and dijkstra_approx.
def exp_c():
    n = 50
    runs = 10
    k = [i for i in range(1, 10)]
    dist_list = []
    dist_approx_list = []
    for k_val in k:
        dist = 0
        dist_approx = 0
        for _ in range(runs):
            G = create_random_complete_graph(n, 100)
            source = random.randint(0, n - 1)
            dist = dist + total_dist(dijkstra(G, source))
            dist_approx = dist_approx + \
                total_dist(dijkstra_approx(G, source, k_val))
        dist_list.append(dist / runs)
        dist_approx_list.append(dist_approx / runs)
        print("k: " + str(k_val) + ", total distance: " + str(dist /
              runs) + ", total distance approx: " + str(dist_approx / runs))

    plt.plot(k, dist_list, label="Dijkstra")
    plt.plot(k, dist_approx_list, label="Dijkstra Approx")
    plt.xlabel("Value of k")
    plt.ylabel("Total distance")
    plt.legend()
    plt.show()


# Fix the value of k of the graph,
# and see how the total distance changes
# as the number of nodes changes for bellman_ford and bellman_ford_approx.
def exp_b():
    k = 3
    runs = 10
    size = [i for i in range(5, 40)]
    dist_list = []
    dist_approx_list = []
    for n in size:
        dist = 0
        dist_approx = 0
        for _ in range(runs):
            G = create_random_complete_graph(n, 100)
            source = random.randint(0, n - 1)
            dist += total_dist(bellman_ford(G, source))
            dist_approx += total_dist(bellman_ford_approx(G, source, k))
        dist_list.append(dist / runs)
        dist_approx_list.append(dist_approx / runs)
        print("n: " + str(n) + ", total distance: " + str(dist / runs) +
              ", total distance approx: " + str(dist_approx / runs))
    plt.plot(size, dist_list, label="Bellman-Ford")
    plt.plot(size, dist_approx_list, label="Bellman-Ford Approx")
    plt.xlabel("Number of nodes")
    plt.ylabel("Total distance")
    plt.legend()
    plt.show()


# Fix the number of nodes of the graph,
# and see how the total distance changes
# as the value of k changes for bellman_ford and bellman_ford_approx.
def exp_d():
    n = 50
    runs = 10
    k = [i for i in range(1, 10)]
    dist_list = []
    dist_approx_list = []
    for k_val in k:
        dist = 0
        dist_approx = 0
        for _ in range(runs):
            G = create_random_complete_graph(n, 100)
            source = random.randint(0, n - 1)
            dist = dist + total_dist(bellman_ford(G, source))
            dist_approx = dist_approx + \
                total_dist(bellman_ford_approx(G, source, k_val))
        dist_list.append(dist / runs)
        dist_approx_list.append(dist_approx / runs)
        print("k: " + str(k_val) + ", total distance: " + str(dist /
              runs) + ", total distance approx: " + str(dist_approx / runs))

    plt.plot(k, dist_list, label="Bellman-Ford")
    plt.plot(k, dist_approx_list, label="Bellman-Ford Approx")
    plt.xlabel("Value of k")
    plt.ylabel("Total distance")
    plt.legend()
    plt.show()
############################################

# Experiment for mystery function


def exp_mys():
    G = DirectedWeightedGraph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(0, 1, 2)
    G.add_edge(1, 3, -3)
    G.add_edge(3, 2, -1)
    G.add_edge(1, 2, 1)

    print(mystery(G))

    sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    times_mystery = []
    times_dijkstra = []
    times_bellman_ford = []

    for n in sizes:
        G = create_random_complete_graph(n, 100)
        start_time = timeit.default_timer()
        mystery(G)
        end_time = timeit.default_timer()
        times_mystery.append(end_time - start_time)
        start_time = timeit.default_timer()
        dijkstra(G, 0)
        end_time = timeit.default_timer()
        times_dijkstra.append(end_time - start_time)
        start_time = timeit.default_timer()
        bellman_ford(G, 0)
        end_time = timeit.default_timer()
        times_bellman_ford.append(end_time - start_time)

    plt.loglog(sizes, times_mystery, label="Mystery")
    plt.loglog(sizes, times_dijkstra, label="Dijkstra")
    plt.loglog(sizes, times_bellman_ford, label="Bellman-Ford")
    plt.xlabel("Number of nodes")
    plt.ylabel("Time")
    plt.legend()
    plt.show()

    # plt.plot(sizes, times_mystery, label="Mystery")
    # plt.plot(sizes, times_dijkstra, label="Dijkstra")
    # plt.plot(sizes, times_bellman_ford, label="Bellman-Ford")
    # plt.xlabel("Number of nodes")
    # plt.ylabel("Time")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    exp_a()
    exp_c()
    exp_b()
    exp_d()
    exp_mys()
