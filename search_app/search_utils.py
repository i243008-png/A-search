import math
import heapq
import itertools
import networkx as nx


def build_graph():
    
    G = nx.Graph()

    coordinates = {
        "Pharmacy": (0, 0),
        "Main_Corridor": (2, 1),
        "Patient_Wing": (1, 4),
        "Nursing_Station": (4, 2),
        "Laboratory": (5, 5),
        "Emergency_Ward": (8, 6),
    }

    edges = [
        ("Pharmacy", "Patient_Wing", 4.1),
        ("Pharmacy", "Main_Corridor", 2.2),
        ("Main_Corridor", "Nursing_Station", 2.2),
        ("Patient_Wing", "Laboratory", 5.0),
        ("Nursing_Station", "Laboratory", 3.2),
        ("Nursing_Station", "Emergency_Ward", 6.0),
        ("Laboratory", "Emergency_Ward", 3.2),
    ]

    for node, pos in coordinates.items():
        G.add_node(node, pos=pos)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    return G, coordinates


def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def heuristic(node, goal, coords):
    return euclidean(coords[node], coords[goal])



def gbfs(graph, coords, start, goal):
    counter = itertools.count()
    frontier = [(heuristic(start, goal, coords), next(counter), start)]
    came_from = {start: None}
    visited = set()
    expansion_order = []

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)
        expansion_order.append(current)

        if current == goal:
            break

        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                if neighbor not in came_from:
                    came_from[neighbor] = current
                heapq.heappush(
                    frontier,
                    (heuristic(neighbor, goal, coords), next(counter), neighbor),
                )

    path = _reconstruct_path(came_from, goal)
    total_cost = _path_cost(graph, path)
    return expansion_order, path, total_cost


def a_star(graph, coords, start, goal, weight_factor=1.0):
    counter = itertools.count()
    h = lambda n: heuristic(n, goal, coords)

    open_set = [(weight_factor * h(start), next(counter), start)]
    g_score = {start: 0.0}
    came_from = {start: None}
    closed = set()
    expansion_order = []
    records = []  # (node, g, h, f) — useful for the g/h/f table

    while open_set:
        f_val, _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        expansion_order.append(current)
        records.append((current, g_score[current], h(current), f_val))

        if current == goal:
            break

        for neighbor in graph.neighbors(current):
            tentative_g = g_score[current] + graph[current][neighbor]["weight"]
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + weight_factor * h(neighbor)
                heapq.heappush(open_set, (f, next(counter), neighbor))

    path = _reconstruct_path(came_from, goal)
    total_cost = _path_cost(graph, path)
    return expansion_order, path, total_cost, records


def _reconstruct_path(came_from, goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()
    return path


def _path_cost(graph, path):
    return sum(
        graph[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
    )