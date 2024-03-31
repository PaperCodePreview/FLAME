import networkx as nx


class Node:
    def __init__(self, panoid, pano_yaw_angle, lat, lng):
        self.panoid = panoid
        self.pano_yaw_angle = pano_yaw_angle
        self.neighbors = {}
        self.coordinate = (lat, lng)

    def get_neighbor_heading(self, panoid):
        for heading, neighbor in self.neighbors.items():
            if neighbor.panoid == panoid:
                return heading
        raise ValueError(f'neighbor "{panoid}" not found for "{self.panoid}"')



class Graph:
    def __init__(self):
        self.nodes = {}
        self.nx_graph = None

    def add_node(self, panoid, pano_yaw_angle, lat, lng):
        self.nodes[panoid] = Node(panoid, pano_yaw_angle, lat, lng)

    def add_edge(self, start_panoid, end_panoid, heading):
        start_node = self.nodes[start_panoid]
        end_node = self.nodes[end_panoid]
        start_node.neighbors[heading] = end_node

    def get_num_neighbors(self, panoid):
        return len(self.nodes[panoid].neighbors)

    def get_target_neighbors(self, panoid):
        return list(nx.all_neighbors(self.nx_graph, panoid))

    def get_shortest_path_length(self, panoid1, panoid2):
        return nx.dijkstra_path_length(self.nx_graph, panoid1, panoid2)

    def get_shortest_path(self, panoid1, panoid2):
        return nx.dijkstra_path(self.nx_graph, panoid1, panoid2)


class GraphLoader:
    def __init__(self, opts):
        self.opts = opts
        self.graph = Graph()
        self.node_file = "%s/graph/nodes.txt" % opts.dataset
        self.link_file = "%s/graph/links.txt" % opts.dataset

    def construct_graph(self):
        with open(self.node_file) as f:
            for line in f:
                panoid, pano_yaw_angle, lat, lng = line.strip().split(',')
                self.graph.add_node(panoid, int(pano_yaw_angle), float(lat), float(lng))

        with open(self.link_file) as f:
            for line in f:
                start_panoid, heading, end_panoid = line.strip().split(',')
                self.graph.add_edge(start_panoid, end_panoid, float(heading))

        G = nx.DiGraph()
        for node in self.graph.nodes.values():
            for neighbor in node.neighbors.values():
                G.add_edge(node.panoid, neighbor.panoid)
        self.graph.nx_graph = G

        return self.graph
