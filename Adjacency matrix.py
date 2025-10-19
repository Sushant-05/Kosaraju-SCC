import pandas as pd
import numpy as np

def csv_to_unweighted_adjacency_matrix(csv_file, has_header=True):
    """
    Convert CSV file to adjacency matrix for unweighted directed graph
    
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Get all edges
    edges = df.values.tolist()
    
    # Extract all unique vertices
    vertices = set()
    for edge in edges:
        if len(edge) >= 2:
            vertices.add(str(edge[0]))
            vertices.add(str(edge[1]))
    
    # Sort vertices and create mapping
    vertices = sorted(list(vertices))
    n = len(vertices)
    vertex_to_index = {v: i for i, v in enumerate(vertices)}
    
    # Initialize adjacency matrix with zeros
    adjacency_matrix = np.zeros((n, n), dtype=int)
    
    # Fill the adjacency matrix
    for edge in edges:
        if len(edge) >= 2:
            source, target = str(edge[0]), str(edge[1])
            i = vertex_to_index[source]
            j = vertex_to_index[target]
            adjacency_matrix[i][j] = 1  # Directed edge
    
    return adjacency_matrix, vertices

def save_adjacency_matrix(matrix, vertices, output_file):
    """
    Save adjacency matrix to CSV file with vertex labels
    
    Args:
        matrix: Adjacency matrix
        vertices: List of vertex names
        output_file: Path for output CSV file
    """
    # Create DataFrame with vertex labels as row and column indices
    df = pd.DataFrame(matrix, index=vertices, columns=vertices)
    
    # Save to CSV
    df.to_csv(output_file)
    print(f"Adjacency matrix saved to {output_file}")

    # Save the adjacency matrix to CSV
    save_adjacency_matrix(matrix, vertices, 'adjacency_matrix.csv')
    
save_adjacency_matrix(*csv_to_unweighted_adjacency_matrix("D:\\Downloads\\Document\\Design and Analysis of Algorithms\\twitter.clean.4k - twitter.clean.4k.csv.csv" , has_header=True),"D:\\Downloads\\Document\\Design and Analysis of Algorithms\\CSV.csv" )
