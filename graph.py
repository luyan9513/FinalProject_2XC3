from typing import Dict
from typing import List

class graph:
    def __init__(self):
        self.adj = {}
        self.weights = {}

    def get_adj_nodes(self, node: int)-> List[int]: 
        return self.adj[node]

    def add_node(self, node: int):
        if node not in self.adj:
            self.adj[node] = []

    def add_edge(self, start: int, end:int, w:float): 
        if start not in self.adj[end]:
            self.adj[start].append(end)
        self.weights[(start, end)] = w

    def get_num_nodes(self) -> int:
        return len(self.adj)
    
    #initial weight 0.0 for all weights
    def w(self, node: int) -> float:
        return 0.0


class Weightedgraph(graph): #weighted graph inherit graph
    
    def __init__(self):
        super().__init__()# use super to call the parent class's __init__ method

    def are_connected(self, node1, node2):   #write a connected method to make sure w method implement correctly
        if node2 in self.adj[node1]:
            return True
        return False
        
    def w(self, node1, node2)-> float:  #override the w
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]


class HeuristicGraph(Weightedgraph): #weighted graph inherit graph
    
    __heuristic = {}    # __ means private
    def __init__(self):
        super().__init__()

    #***The set method to set the __heuristic dict for HeuristicGraph 
    def set_heuristic(self, h):
        self.__heuristic=h

    def get_heuristic(self) -> Dict[int, float]:
        return self.__heuristic

