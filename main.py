from utils import parse_args, Stocker

def main():
    args = parse_args()
    stock = Stocker(args)
    stock.pre_process()
    stock.train()
    stock.test()
    stock.plot()

if __name__ == '__main__':
    main()