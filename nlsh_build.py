from data_parser import BuildParser, read_mnist, read_sift

def main():
    p = BuildParser(sys.argv)

    # open files
    if p.type == "mnist":
        data = read_mnist(p.input)
    else:
        data = read_sift(p.input)

    print("Dataset loaded.")


if __name__ == "__main__":
    main()
