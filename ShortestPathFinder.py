import SPAlgorithm
import graph


class ShortestPathFinder:
    def __init__(self):
        self.algorithm = None
        self.graph = None

    def calc_short_path(self, source: int, dest: int) -> float:
        return self.algorithm.calc_sp(self.graph, source, dest)

    def set_graph(self, graph: graph):
        self.graph = graph

    def set_algorithm(self, algorithm: SPAlgorithm):
        self.algorithm = algorithm


def test():
    s = ShortestPathFinder()
    s.set_algorithm(SPAlgorithm.Bellman_Ford)
    s.set_graph(graph.Weightedgraph())

    for node in range(0, 7):
        s.graph.add_node(node)

    s.graph.add_edge(0, 1, 4)
    s.graph.add_edge(0, 2, 3)
    s.graph.add_edge(1, 4, 2)
    s.graph.add_edge(2, 3, 6)
    s.graph.add_edge(2, 5, 1)
    s.graph.add_edge(3, 6, 8)
    s.graph.add_edge(4, 6, 6)
    s.graph.add_edge(5, 6, 3)

    print(s.calc_short_path(0, 6))


test()


def test2():
    s = ShortestPathFinder()
    s.set_algorithm(SPAlgorithm.A_Star_Adapter)
    s.set_graph(graph.HeuristicGraph())

    for node in range(0, 7):
        s.graph.add_node(node)

    s.graph.add_edge(0, 1, 4)
    s.graph.add_edge(0, 2, 3)
    s.graph.add_edge(1, 4, 2)
    s.graph.add_edge(2, 3, 6)
    s.graph.add_edge(2, 5, 1)
    s.graph.add_edge(3, 6, 8)
    s.graph.add_edge(4, 6, 6)
    s.graph.add_edge(5, 6, 3)
    h = {0: 6.0, 1: 4.0, 2: 2.0, 3: 1.0, 4: 2.0, 5: 3.0, 6: 0.0}
    s.graph.set_heuristic(h)
    print(s.calc_short_path(0, 6))
    print(s.graph.get_heuristic())


test2()
