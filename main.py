import pandas as pd
import numpy as np
from utils import pre_process, parse_args, evaluate
from catboost import CatBoostRegressor

def main(args):
    x_train, y_train = pre_process(path=args.train_path, window_size=args.window_size)
    model = CatBoostRegressor()
    model.fit(x_train, y_train)

    x_test, y_test = pre_process(path=args.test_path, window_size=args.window_size)
    y_pred = model.predict(x_test)
    evaluate(y_test, y_pred)

if __name__ == '__main__':
    args = parse_args()
    main(args)