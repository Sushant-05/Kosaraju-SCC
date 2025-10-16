import pandas as pd
import numpy as np

def csv_to_unweighted_adjacency_matrix(csv_file, has_header=True):
    """
    Convert CSV file to adjacency matrix for unweighted directed graph
    
    Args:
        csv_file (str): Path to CSV file
        has_header (bool): Whether CSV has header row
    
    Returns:
        tuple: (adjacency_matrix, vertices)
    """
    # Read CSV file
    if has_header:
        df = pd.read_csv(csv_file)
    else:
        df = pd.read_csv(csv_file, header=None)
    
    # Get all edges
    edges = df.values.tolist()
    
    # Extract all unique vertices
    vertices = set()
    for edge in edges:
        if len(edge) >= 2:
            vertices.add(edge[0])
            vertices.add(edge[1])
    
    # Sort vertices and create mapping
    vertices = sorted(list(vertices))
    n = len(vertices)
    vertex_to_index = {v: i for i, v in enumerate(vertices)}
    
    # Initialize adjacency matrix with zeros
    adjacency_matrix = np.zeros((n, n), dtype=int)
    
    # Fill the adjacency matrix
    for edge in edges:
        if len(edge) >= 2:
            source, target = edge[0], edge[1]
            i = vertex_to_index[source]
            j = vertex_to_index[target]
            adjacency_matrix[i][j] = 1  # Directed edge
    
    return adjacency_matrix, vertices

# Example usage
if __name__ == "__main__":
    # Example CSV data for unweighted directed graph
    csv_data = """source,target
    A,B
    A,C
    B,D
    C,D
    D,A"""
    
    with open('unweighted_graph.csv', 'w') as f:
        f.write(csv_data)
    
    matrix, vertices = csv_to_unweighted_adjacency_matrix('unweighted_graph.csv')
    print("Vertices:", vertices)
    print("Adjacency Matrix:")
    print(matrix)
