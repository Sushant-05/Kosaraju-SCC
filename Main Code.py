import collections
import random
import time
import tracemalloc
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(6000)

class Graph:
    def __init__(self, vertices):
        """
        Initialize graph with given number of vertices.
        """
        self.V = vertices
        # Adjacency list for original graph
        self.graph = [[] for _ in range(vertices)]
        # Adjacency list for transpose graph
        self.transpose = [[] for _ in range(vertices)]
    
    def add_edge(self, u, v):
        """
        Add directed edge from vertex u to vertex v.
        
        Args:
            u (int): Source vertex
            v (int): Destination vertex
        """
        self.graph[u].append(v)
    
    def build_transpose(self):
        """
        Build the transpose graph by reversing all edges.
        For every edge u->v in original graph, add edge v->u in transpose.
        """
        for u in range(self.V):
            for v in self.graph[u]:
                self.transpose[v].append(u)
    
    def dfs_first(self, v, visited, stack):
        """
        First DFS pass: Perform DFS on original graph and fill stack with vertices
        in order of decreasing finishing times.
        
        Args:
            v (int): Current vertex
            visited (list): Track visited vertices
            stack (list): Stack to store vertices by finishing time
        """
        visited[v] = True
        for neighbor in self.graph[v]:
            if not visited[neighbor]:
                self.dfs_first(neighbor, visited, stack)
        stack.append(v)
    
    def dfs_second(self, v, visited, component, graph_type='transpose'):
        """
        Second DFS pass: Perform DFS on transpose graph to find SCC.
        
        Args:
            v (int): Current vertex
            visited (list): Track visited vertices
            component (list): Current strongly connected component
            graph_type (str): Which graph to traverse ('transpose' or 'original')
        """
        visited[v] = True
        component.append(v)
        
        # Choose which graph to traverse
        graph_to_use = self.transpose if graph_type == 'transpose' else self.graph
        
        for neighbor in graph_to_use[v]:
            if not visited[neighbor]:
                self.dfs_second(neighbor, visited, component, graph_type)
    
    def dfs_first_iterative(self, v, visited, stack):
        """
        Iterative version of first DFS pass to avoid recursion limits for large graphs.
        
        Args:
            v (int): Starting vertex
            visited (list): Track visited vertices
            stack (list): Stack to store vertices by finishing time
        """
        dfs_stack = [v]
        visited[v] = True
        
        # To track when we finish processing a vertex
        finish_order = []
        
        while dfs_stack:
            current = dfs_stack[-1]
            
            # Find unvisited neighbor
            found_unvisited = False
            for neighbor in self.graph[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    dfs_stack.append(neighbor)
                    found_unvisited = True
                    break
            
            # If no unvisited neighbors, we've finished this vertex
            if not found_unvisited:
                finished_vertex = dfs_stack.pop()
                stack.append(finished_vertex)
    
    def kosaraju_scc(self, use_iterative=False):
        """
        Main function to find all Strongly Connected Components using Kosaraju's algorithm.
        
        Args:
            use_iterative (bool): Use iterative DFS to avoid recursion limits
            
        Returns:
            list: List of SCCs, where each SCC is a list of vertices
        """
        # Step 1: First DFS pass on original graph
        stack = []
        visited = [False] * self.V
        
        for i in range(self.V):
            if not visited[i]:
                if use_iterative:
                    self.dfs_first_iterative(i, visited, stack)
                else:
                    self.dfs_first(i, visited, stack)
        
        # Step 2: Build transpose graph
        self.build_transpose()
        
        # Step 3: Second DFS pass on transpose graph
        visited = [False] * self.V
        scc_list = []
        
        while stack:
            v = stack.pop()
            if not visited[v]:
                component = []
                self.dfs_second(v, visited, component, 'transpose')
                scc_list.append(component)
        
        return scc_list
    
    def validate_scc(self, scc_list):
        """
        Validate that each component is indeed strongly connected.
        
        Args:
            scc_list (list): List of SCCs found by the algorithm
            
        Returns:
            bool: True if all components are valid SCCs
        """
        for component in scc_list:
            if len(component) > 1:
                # For each pair of vertices in the component, check mutual reachability
                for i in range(len(component)):
                    for j in range(i + 1, len(component)):
                        if not (self.is_reachable(component[i], component[j]) and 
                                self.is_reachable(component[j], component[i])):
                            return False
        return True
    
    def is_reachable(self, start, end):
        """
        Check if there's a path from start to end using BFS.
        
        Args:
            start (int): Starting vertex
            end (int): Target vertex
            
        Returns:
            bool: True if end is reachable from start
        """
        if start == end:
            return True
            
        visited = [False] * self.V
        queue = collections.deque([start])
        visited[start] = True
        
        while queue:
            current = queue.popleft()
            for neighbor in self.graph[current]:
                if neighbor == end:
                    return True
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return False

def generate_random_graph(vertices, edge_density=0.3):
    """
    Generate a random directed graph.
    
    Args:
        vertices (int): Number of vertices
        edge_density (float): Probability of having an edge between any two vertices
        
    Returns:
        Graph: Randomly generated graph
    """
    graph = Graph(vertices)
    
    for u in range(vertices):
        for v in range(vertices):
            if u != v and random.random() < edge_density:
                graph.add_edge(u, v)
    
    return graph

def generate_graph_with_scc(vertices, scc_sizes):
    """
    Generate a graph with known SCC structure for testing.
    
    Args:
        vertices (int): Total number of vertices
        scc_sizes (list): List of sizes for each SCC
        
    Returns:
        Graph: Graph with specified SCC structure
    """
    graph = Graph(vertices)
    vertex_index = 0
    
    # Create strongly connected components
    for size in scc_sizes:
        # Add edges to make this component strongly connected
        component_vertices = list(range(vertex_index, vertex_index + size))
        
        # Create a cycle to ensure strong connectivity
        for i in range(len(component_vertices)):
            u = component_vertices[i]
            v = component_vertices[(i + 1) % len(component_vertices)]
            graph.add_edge(u, v)
        
        vertex_index += size
    
    # Add some random edges between components
    for u in range(vertices):
        for v in range(vertices):
            if u != v and random.random() < 0.1:
                graph.add_edge(u, v)
    
    return graph

def run_experiments():
    """
    Run comprehensive experiments as specified in the project requirements.
    """
    print("=== Kosaraju's Algorithm - SCC Detection Experiments ===\n")
    
    # Experiment 1: Scalability Analysis
    print("Experiment 1: Scalability Analysis")
    print("=" * 50)
    
    sizes = [100, 500, 1000, 2000, 5000]
    results = []
    
    for size in sizes:
        print(f"Testing with {size} vertices...")
        
        # Generate random graph
        graph = generate_random_graph(size, edge_density=0.2)
        
        # Measure runtime
        start_time = time.time()
        scc_list = graph.kosaraju_scc(use_iterative=size > 1000)
        end_time = time.time()
        
        runtime = end_time - start_time
        
        # Measure memory
        tracemalloc.start()
        scc_list = graph.kosaraju_scc(use_iterative=size > 1000)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_mb = peak / 10**6  # Convert to MB
        
        # Count SCCs and find largest
        scc_count = len(scc_list)
        largest_scc = max(len(component) for component in scc_list) if scc_list else 0
        
        results.append({
            'vertices': size,
            'edges': sum(len(adj) for adj in graph.graph),
            'runtime': runtime,
            'memory': memory_mb,
            'scc_count': scc_count,
            'largest_scc': largest_scc
        })
        
        print(f"  Runtime: {runtime:.4f}s, Memory: {memory_mb:.2f}MB, SCCs: {scc_count}")
    
    # Experiment 2: Correctness Validation
    print("\nExperiment 2: Correctness Validation")
    print("=" * 50)
    
    # Test with known structure
    test_graph = generate_graph_with_scc(10, [3, 3, 4])
    scc_list = test_graph.kosaraju_scc()
    
    print("Known structure test:")
    print(f"Generated SCCs: {scc_list}")
    print(f"Validation: {test_graph.validate_scc(scc_list)}")
    
    # Manual verification for small graph
    small_graph = Graph(5)
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3)]  # SCCs: [0,1,2] and [3,4]
    for u, v in edges:
        small_graph.add_edge(u, v)
    
    scc_list = small_graph.kosaraju_scc()
    print(f"\nSmall graph test:")
    print(f"Expected SCCs: [[0,1,2], [3,4]]")
    print(f"Found SCCs: {scc_list}")
    print(f"Validation: {small_graph.validate_scc(scc_list)}")
    
    # Experiment 3: Real-world Pattern Simulation
    print("\nExperiment 3: Real-world Pattern Simulation")
    print("=" * 50)
    
    # Simulate web graph pattern: one giant SCC + many small components
    web_like_graph = Graph(1000)
    
    # Create giant SCC (vertices 0-299)
    for i in range(300):
        web_like_graph.add_edge(i, (i + 1) % 300)
        web_like_graph.add_edge((i + 1) % 300, i)
    
    # Add many small components and random edges
    for i in range(300, 1000, 2):
        if i + 1 < 1000:
            web_like_graph.add_edge(i, i + 1)
            web_like_graph.add_edge(i + 1, i)
    
    # Add some random cross edges
    for _ in range(500):
        u = random.randint(0, 999)
        v = random.randint(0, 999)
        if u != v:
            web_like_graph.add_edge(u, v)
    
    scc_list = web_like_graph.kosaraju_scc(use_iterative=True)
    scc_sizes = [len(component) for component in scc_list]
    scc_sizes.sort(reverse=True)
    
    print(f"Web-like graph SCC size distribution:")
    print(f"Top 5 SCC sizes: {scc_sizes[:5]}")
    print(f"Total SCCs: {len(scc_list)}")
    
    return results

def plot_results(results):
    """
    Plot experimental results.
    
    Args:
        results (list): Experimental results from run_experiments()
    """
    vertices = [r['vertices'] for r in results]
    runtimes = [r['runtime'] for r in results]
    memory_usage = [r['memory'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Runtime plot
    ax1.plot(vertices, runtimes, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Vertices')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Runtime vs Graph Size')
    ax1.grid(True, alpha=0.3)
    
    # Memory usage plot
    ax2.plot(vertices, memory_usage, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Vertices')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage vs Graph Size')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kosaraju_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


print("Kosaraju's Algorithm - Strongly Connected Components Detection")
print("=" * 60)
    
# Create a sample graph
g = Graph(8)
edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3), (5, 6), (6, 7), (7, 6)]
for u, v in edges:
    g.add_edge(u, v)
    
print("Graph edges:", edges)
    
# Find SCCs
scc_list = g.kosaraju_scc()
    
print("\nStrongly Connected Components:")
for i, component in enumerate(scc_list):
    print(f"SCC {i + 1}: {component}")
    
# Run comprehensive experiments
results = run_experiments()
    
# Plot results
plot_results(results)

