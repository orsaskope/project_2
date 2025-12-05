import numpy as np
import time

# ---------------------------------------------------
# THESE ARE USED WHEN BRUTE FORCE IS CHOSEN AS THE WAY OF BUILDING GRAPH (Further down are the ivfflat ones too)
# ---------------------------------------------------

# ---------------------------------------------------
# Brute-force kNN
# ---------------------------------------------------
def compute_knn(dataset: np.ndarray, k: int):
    N, D = dataset.shape
    knn_graph = [[] for _ in range(N)]

    for i in range(N):
        start_time = time.time()

        diff = dataset - dataset[i]
        dists = np.linalg.norm(diff, axis=1)
        dists[i] = np.inf  # prevent self-match

        idx = np.argpartition(dists, k)[:k]
        top_k = sorted([(dists[j], j) for j in idx])

        knn_graph[i] = [j for (_, j) in top_k]

        print(f"[INFO] Vector {i}: kNN in {time.time() - start_time:.4f}s")

    return knn_graph


# ---------------------------------------------------
# Build weighted kNN graph
# ---------------------------------------------------
def build_weighted_knn_graph(knn_neighbors):
    N = len(knn_neighbors)
    w_knn = {i: {} for i in range(N)}

    for i in range(N):
        for j in knn_neighbors[i]:
            if i == j:
                continue

            # Determine mutual (weight=2) or one-sided (weight=1)
            weight = 2 if i in knn_neighbors[j] else 1

            # Insert BOTH directions to make graph undirected
            w_knn[i][j] = weight
            w_knn[j][i] = weight

    return w_knn


# ---------------------------------------------------
# Convert weighted graph to CSR
# ---------------------------------------------------
def knn_graph_to_csr(w_knn_graph):
    N = len(w_knn_graph)
    xadj = [0]
    adjncy = []
    adjcwgt = []
    vwgt = [1] * N

    for i in range(N):
        neighbors = w_knn_graph[i]
        for j, w in neighbors.items():
            adjncy.append(j)
            adjcwgt.append(w)
        xadj.append(len(adjncy))

    return xadj, adjncy, adjcwgt, vwgt



# ---------------------------------------------------
# THESE ARE USED WHEN IVFFLAT IS CHOSEN AS THE WAY OF BUILDING GRAPH 
# ---------------------------------------------------

# ---------------------------------------------------
# Build weighted kNN graph (IVFFLAT)
# ---------------------------------------------------

def bwg_ivfflat(path, N):
    graph = {}  #ID: neighbour list
    fd = open(path, "r")
    for line in fd:
        line = line.strip()
        # Get every neighbour
        if line.startswith("NODE"):
            parts = line.split(":") # Splits the line in ":" since the format of the output file is nodeID:Neighbours
            left = parts[0]
            right = parts[1]

            qid = int(left.split()[1]) # Takes the query id ignoring the word "NODE"
            neighs_str = right.strip().split()
            neighs = []
            # Append each neighbour
            for x in neighs_str:
                neighs.append(int(x))

            graph[qid] = neighs
    fd.close()

    print("\n=== Parsed graph ===")
    # for q, neighs in graph.items():
    #     print(f"Query {q} -> {neighs}")

    weighted = {} # A dictionary of dictionaries
    for q in graph:
        for neighbor in graph[q]:
            if neighbor == q:   continue

            # If vectors are not in graph, create a dictionary to keep their neighbours w the weights
            if q not in weighted:
                weighted[q] = {}
            if neighbor not in weighted:
                weighted[neighbor] = {}

            if q in weighted[neighbor]:
                weighted[neighbor][q] += 1
                weighted[q][neighbor] += 1
            else:
                weighted[neighbor][q] = 1
                weighted[q][neighbor] = 1

    # Make sure all nodes are in weighted graph
    for i in range(N):
        if i not in weighted:
            print("found missing node:", i)
            weighted[i] = {}
    return weighted

# ---------------------------------------------------
# Convert weighted graph to CSR (IVFFLAT)
# ---------------------------------------------------

def build_csr_ivfflat(graph, N):
    xadj = [0]  # Offset: were node's neighbours are in adjncy. (xadj[i+1]: end index)
    adjncy = [] # adjacency list
    adjcwgt = [] # adjacency weight
    vwgt = [1]*N   # nodes weight

    for i in range(N):
        neighs = list(graph[i].keys())
        for n in neighs:
            adjncy.append(n)
            adjcwgt.append(graph[i][n])
        
        xadj.append(len(adjncy))

    print("xadj: %d adjncy: %d adjcwgt: %d vwgt: %d", len(xadj), len(adjncy), len(adjcwgt), len(vwgt))

    return xadj, adjncy, adjcwgt, vwgt
