from utils import pre_process, parse_args, evaluate
from catboost import CatBoostRegressor

def main(args):
    x_train, x_test, y_train, y_test = pre_process(
        path=args.data_path, 
        window_size=args.window_size,
        test_size=args.test_size,
        )
    model = CatBoostRegressor(verbose=100)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    evaluate(y_test, y_pred)

if __name__ == '__main__':
    args = parse_args()
    main(args)