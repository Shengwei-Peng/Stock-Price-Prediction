import pandas as pd
import numpy as np
from utils import sliding_window, parse_args

def main(args):
    train_data = pd.read_csv(args.train_file)
    x_train, y_train = sliding_window(data=train_data, window_size=args.window_size)

if __name__ == '__main__':
    args = parse_args()
    main(args)