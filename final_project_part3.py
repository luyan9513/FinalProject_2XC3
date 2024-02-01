import csv
import math
import timeit
import min_heap
import matplotlib.pyplot as plot
from collections import defaultdict


class DirectedWeightedGraphForStation:

    def __init__(self):
        self.adj = {}
        self.weights = {}
        self.lines = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight, line=None):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight
        if line is not None:
            self.lines[(node1, node2)] = line

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)


def dijkstra_for_part3(G, source, connections):
    pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {}  # Distance dictionary
    lines_used = {}
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
                # Get the line used for this connection
                line = connections[(current_node, neighbour)]
                Q.decrease_key(
                    neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + \
                    G.w(current_node, neighbour)
                pred[neighbour] = current_node
                lines_used[neighbour] = line
    return dist, pred, lines_used  # ********* return pred for test purpose


def a_star_for_part3(G, s, d, h):
    pred = {}
    dist = {}
    lines_used = {}
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
            new_dist = dist[current_node] + G.w(current_node, neighbour)
            if new_dist < dist[neighbour]:
                Q.decrease_key(neighbour, new_dist + h[neighbour])
                dist[neighbour] = new_dist
                pred[neighbour] = current_node

                if (current_node, neighbour) in G.lines:
                    line = G.lines[(current_node, neighbour)]
                    lines_used[neighbour] = line

    return pred, lines_used, dist[d]


def haversine_distance(lat1, lon1, lat2, lon2):
    # Earth radius
    R = 6371

    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2) * math.sin(d_lon / 2))

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance


stations = {}

with open('london_stations.csv', 'r') as csvfile1:
    csvreader1 = csv.reader(csvfile1)
    # Skip header
    next(csvreader1)

    for row in csvreader1:
        station_id, lat, lon = int(row[0]), float(row[1]), float(row[2])
        stations[station_id] = (lat, lon)

london_subway_graph = DirectedWeightedGraphForStation()

for station_id in stations.keys():
    london_subway_graph.add_node(station_id)

connections = {}

with open('london_connections.csv', 'r') as csvfile2:
    csvreader2 = csv.reader(csvfile2)
    # Skip header
    next(csvreader2)

    for row in csvreader2:
        station1, station2, line = int(row[0]), int(row[1]), int(row[2])
        lat1, lon1 = stations[station1]
        lat2, lon2 = stations[station2]
        distance = haversine_distance(lat1, lon1, lat2, lon2)

        london_subway_graph.add_edge(station1, station2, distance, line)
        london_subway_graph.add_edge(station2, station1, distance, line)

        connections[(station1, station2)] = line
        connections[(station2, station1)] = line

s = 1
# d=303
h = {}

for node in london_subway_graph.adj.keys():
    if london_subway_graph.adj[node]:
        h[node] = min(london_subway_graph.w(node, neighbor)
                      for neighbor in london_subway_graph.adj[node])
    else:
        h[node] = 0


def total_distance_by_line(lines_used, graph, pred):
    distance_by_line = defaultdict(float)
    for node, line in lines_used.items():
        parent_node = pred[node]
        distance_by_line[line] += graph.w(parent_node, node)
    return distance_by_line


def experiment1(t):

    times1 = []
    for s in range(1, 303):
        if s == 189:
            continue
        for d in range(s+1, 303):
            if d == 189:
                continue
            for _ in range(t):
                total1 = 0
                start1 = timeit.default_timer()
                pred, line, dist_d = a_star_for_part3(
                    london_subway_graph, s, d, h)
                end1 = timeit.default_timer()
                total1 += end1 - start1
            average = total1 / t
            times1.append(average)
            if average < 0.001 or average > 0.003:
                print(f"{s},{d}uses {total1} s")
    return times1


def experiment2(t):
    total2 = 0
    times2 = []
    for _ in range(t):
        start2 = timeit.default_timer()
        ppred, ddist, lines_used = dijkstra_for_part3(
            london_subway_graph, s, connections)
        end2 = timeit.default_timer()
        total2 += end2 - start2
        times2.append(end2 - start2)
    print(f"dijkstra uses {total2} s")
    return times2


def experiment3():
    dist, pred, lines_used = dijkstra_for_part3(
        london_subway_graph, s, connections)
    distance_by_line = total_distance_by_line(
        lines_used, london_subway_graph, pred)

    for line, distance in distance_by_line.items():
        print(f"Distance in line {line}: {distance} km")


experiment_times = 1
result1 = experiment1(experiment_times)
# result2 = experiment2(experiment_times)
plot.plot(result1, label='a_star')
# plot.plot(result2, label='dijkstra')
plot.xlabel('Number')
plot.ylabel('Time')
plot.legend()
plot.show()
