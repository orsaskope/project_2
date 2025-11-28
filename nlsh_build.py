import sys
import os
import subprocess
from data_parser import BuildParser, read_mnist, read_sift

def main():
    p = BuildParser(sys.argv)

    # open files
    if p.type == "mnist":
        data = read_mnist(p.input)
    else:
        data = read_sift(p.input)

    print("Dataset loaded.")
    exec_path = "../../PRJ1/project_1/search"
    result = subprocess.run(
    [exec_path, "-ivfflat",
    "-type", "mnist",
    "-seed", "9",
    "-d", "input.dat",
    "-q", "query.dat",
    "-kclusters", "4",
    "-range", "true",
    "-N", str(p.knn),
    "-o", "tmp.txt",
    "-nprobe", "2",
    "-sample_pq", "true",
    "-R", "500"],
    capture_output=True,
    text=True)
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    print("CODE:", result.returncode)

    output_path = "../../PRJ1/project_1/output.txt"
    graph = read_knn_output(output_path)

# Read first project's output to build the graph
def read_knn_output(path):
    graph = {}  # QueryID: neighbour list
    fd = open(path, "r")
    for line in fd:
        line = line.strip()
        # Get every query's neighbours
        if line.startswith("Query"):
            parts = line.split(":") # Splits the line in ":" since the format of the output file is QueryID:Neighbours
            left = parts[0]
            right = parts[1]

            qid = int(left.split()[1]) # Takes the query id ignoring the word "Query"
            neighs_str = right.strip().split()
            neighs = []
            # Append each neighbour
            for x in neighs_str:
                neighs.append(int(x))

            graph[qid] = neighs
    fd.close()

    print("\n=== Parsed graph ===")
    for q, neighs in graph.items():
        print(f"Query {q} -> {neighs}")

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
                weighted[neighbor][q] = 2
                weighted[q][neighbor] = 2
            else:
                weighted[neighbor][q] = 1
                weighted[q][neighbor] = 1
    return weighted

if __name__ == "__main__":
    main()
