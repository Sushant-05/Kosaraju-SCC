# Kosaraju's Algorithm Implementation for SCC detection
This project implements Kosaraju's algorithm for detecting Strongly Connected Components (SCCs) in directed graphs. Strongly Connected Components are fundamental structures in graph theory that represent maximal subgraphs where every vertex is reachable from every other vertex. The algorithm employs a two-pass Depth-First Search (DFS) approach with linear time complexity $O(V + E)$, making it highly efficient for large-scale graph analysis. We present the theoretical foundation, implementation details, complexity analysis, and experimental results on both synthetic and real-world datasets. The algorithm demonstrates excellent scalability and correctness across various graph structures.
## 📋 Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quickstart)
- [Usage](#usage)
- [Example](#example)


## 🚀 Overview

This project implements **Kosaraju's algorithm** for finding **Strongly Connected Components (SCCs)** in directed graphs. SCCs are fundamental structures in graph theory where every vertex is reachable from every other vertex within the component.

**Key Applications:**
- Web page clustering and analysis
- Social network community detection
- Software dependency cycle detection
- Circuit design and feedback loop analysis


## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
  libraries:-
- matplotlib
- pandas
- sys
- tracemalloc
- time
- random
- collection

## Quick Start
git clone https://github.com/yourusername/kosaraju-scc.git
cd kosaraju-scc
pip install -r requirements.txt
python main.py

## Usage
Download "kosaraju.py" and "Adjacency matrix.py" from the repository.
================================================================================================Basic Usage
from kosaraju import Graph

# Create a graph
g = Graph(5)
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 0)  # Creates SCC: {0, 1, 2}
g.add_edge(2, 3)
g.add_edge(3, 4)
g.add_edge(4, 3)  # Creates SCC: {3, 4}

# Find SCCs
scc_list = g.kosaraju_scc()
print("Strongly Connected Components:", scc_list)
# Output: [[0, 1, 2], [3, 4]]

=================================================================================================Using from command line interface
# Run on a sample graph
python main.py --vertices 100 --density 0.2

# Test with specific graph file
python main.py --input graph.txt --output results.txt

# Run performance benchmarks
python benchmarks.py --sizes 100 500 1000 5000
=================================================================================================Advanced Usage
# For large graphs, use iterative DFS
scc_list = g.kosaraju_scc(use_iterative=True)

# Validate SCC correctness
is_valid = g.validate_scc(scc_list)

# Generate performance report
from experiments import run_experiments
results = run_experiments()


## Example
### Simple Graph
# Graph: 0→1→2→0, 2→3, 3→4→3
g = Graph(5)
edges = [(0,1), (1,2), (2,0), (2,3), (3,4), (4,3)]
for u, v in edges:
    g.add_edge(u, v)

scc_list = g.kosaraju_scc()
print(scc_list)  # [[0, 1, 2], [3, 4]]
