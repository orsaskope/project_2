import sys
import os
import subprocess
import kahip
from train_utils import train_model_cpu
from data_parser import BuildParser, read_mnist, read_sift

def main():
    p = BuildParser(sys.argv)

    # open files
    if p.type == "mnist":
        data = read_mnist(p.input)
        N = data.number_of_images
    else:
        data = read_sift(p.input)
        N = data.count

    print("Dataset loaded.")
    # exec_path = "./search"
    # result = subprocess.run(
    # [exec_path, "-ivfflat",
    # "-type", "mnist",
    # "-seed", "9",
    # "-d", "input.dat",
    # "-kclusters", "4",
    # "-range", "false",
    # "-N", str(p.knn),
    # "-o", "tmp.txt",
    # "-nprobe", "2",
    # "-R", "500"],
    # capture_output=True,
    # text=True)
    # print("STDOUT:\n", result.stdout)

    output_path = "./tmp.txt"
    graph = build_weighted_graph(output_path, N)
    xadj, adjncy, adjcwgt, vwgt = build_csr(graph, N)

    edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy, p.m, p.imbalance, False, p.seed, p.kahip_mode)
    model = train_model_cpu(
    X,
    y,
    nodes=p.nodes,
    layers=p.layers,
    m=p.m,
    batch=p.batch_size,
    epochs=p.epochs,
    lr=p.lr
)


# Read first project's output to build the graph
def build_weighted_graph(path, N):
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

def build_csr(graph, N):
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

if __name__ == "__main__":
    main()
