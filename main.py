from utils import parse_args, Stocker

def main():
    args = parse_args()
    stock = Stocker(args)
    stock.preprocess()
    stock.train()
    stock.evaluate()
    stock.visualize()

if __name__ == '__main__':
    main()