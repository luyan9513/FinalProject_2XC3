import min_heap
import graph


class SPAlgorithm:  # SPAlgorithm interfaces
    def calc_sp(graph: graph, source: int, dest: int) -> float:
        return 0.0


class Dijkstra(SPAlgorithm):  # Dijkstra inherit SPAlgorithm
    def calc_sp(graph: graph.Weightedgraph(), source: int, dest: int) -> float:
        pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
        dist = {}  # Distance dictionary
        Q = min_heap.MinHeap([])
        nodes = list(graph.adj.keys())

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
            for neighbour in graph.adj[current_node]:
                if dist[current_node] + graph.w(current_node, neighbour) < dist[neighbour]:
                    Q.decrease_key(
                        neighbour, dist[current_node] + graph.w(current_node, neighbour))
                    dist[neighbour] = dist[current_node] + \
                        graph.w(current_node, neighbour)
                    pred[neighbour] = current_node
                if neighbour == dest:
                    break
        return float(dist[dest])


class Bellman_Ford(SPAlgorithm):  # Bellman_Ford inherit SPAlgorithm
    def calc_sp(graph: graph.Weightedgraph(), source: int, dest: int) -> float:
        pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
        dist = {}  # Distance dictionary
        nodes = list(graph.adj.keys())

        # Initialize distances
        for node in nodes:
            dist[node] = float("inf")
        dist[source] = 0

        # Meat of the algorithm
        for _ in range(len(graph.adj)):
            for node in nodes:
                for neighbour in graph.adj[node]:
                    if dist[neighbour] > dist[node] + graph.w(node, neighbour):
                        dist[neighbour] = dist[node] + graph.w(node, neighbour)
                        pred[neighbour] = node
                    if neighbour == dest:
                        break
        return float(dist[dest])


class A_Star_Adapter(SPAlgorithm):  # adapter for A_Star

    # calc the sp for the given HeuristicGraph using the A_Star class
    def calc_sp(graph: graph.HeuristicGraph(), source: int, dest: int) -> float:
        # Call the A_Star algorithm
        a_star = A_Star(graph)
        result = a_star.run_a_star(source, dest)
        spd = result[1]
        return float(spd)


class A_Star:
    def __init__(self, graph):
        self.graph = graph

    def run_a_star(self, s, d):
        pred = {}
        dist = {}
        G = self.graph
        h = self.graph.get_heuristic()
        Q = min_heap.MinHeap([])
        nodes = list(G.adj.keys())

        for node in nodes:
            Q.insert(min_heap.Element(node, float("inf")))
            dist[node] = float("inf")
        Q.decrease_key(s, h[s])
        dist[s] = 0

        while not Q.is_empty():
            current_element = Q.extract_min()
            current_node = current_element.value
            if current_node == d:
                break
            for neighbour in G.adj[current_node]:
                if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                    Q.decrease_key(
                        neighbour, dist[current_node] + G.w(current_node, neighbour) + h[neighbour])
                    dist[neighbour] = dist[current_node] + \
                        G.w(current_node, neighbour)
                    pred[neighbour] = current_node

        return pred, dist[d]
