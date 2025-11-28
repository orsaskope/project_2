from data_parser import SearchParser, read_mnist, read_sift

def main():
    p = SearchParser(sys.argv)

    if p.type == "mnist":
        data = read_mnist(p.input)
        queries = read_mnist(p.query)
    else:
        data = read_sift(p.input)
        queries = read_sift(p.query)

    print("Dataset + queries loaded")

if __name__ == "__main__":
    main()