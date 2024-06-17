# Stock-Price-Prediction

This repository contains code and resources for predicting stock prices using machine learning techniques. The goal is to develop a model that can accurately predict future stock prices based on historical data.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Stock price prediction is a challenging task due to the volatility and complexity of financial markets. This project aims to leverage machine learning algorithms to make accurate predictions based on historical stock price data.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Shengwei0516/Stock-Price-Prediction.git
    cd Stock-Price-Prediction
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Data

The dataset used in this project consists of historical stock prices fetched using the `twstock` library. Here is an example of how to fetch the data:

```python
import twstock

# Fetch historical stock data for stock code '2330' from January 2014 to present
stock = twstock.Stock('2330')
stock.fetch_from(2014, 1)
```

Below is a brief description of each column in the dataset and a sample table to illustrate the data format:

| Column      | Description                                   | Data Type  |
| ----------- | --------------------------------------------- | ---------- |
| Date        | The trading date                              | YYYY-MM-DD |
| Capacity    | The total traded volume of stocks             | int        |
| Turnover    | The total value of traded stocks              | int        |
| Open        | The opening price of the stock                | float      |
| High        | The highest price during the day              | float      |
| Low         | The lowest price during the day               | float      |
| Close       | The closing price of the stock                | float      |
| Change      | The price change compared to the previous day | float      |
| Transaction | The number of transactions                    | int        |

## Usage

To use the stock price prediction tool, follow these steps:

1. **Prepare your dataset**: Ensure your dataset is in the correct format and located in the `data` directory.
2. **Run the main script**: Execute the following command to start the process:
   ```bash
   python main.py --data_path data/2330.csv --model lstm --test_size 20 --window_size 20
   ```
The script will perform the following steps:
   - **Preprocessing**: Clean and prepare the data for analysis.
   - **Training**: Train the prediction models on the prepared data.
   - **Evaluation**: Assess the performance of the trained models and generate performance metrics.
   - **Visualization**: Create visualizations to compare the predicted stock prices with actual prices.

### Configuration Options
You can customize the behavior of the tool using various configuration options. Here are some examples:

- `data_path`: Specify the path to the input data file.
- `model`: Choose the machine learning model to use (e.g., CatBoost.).
- `test_size`: Specify the number of latest trading days to use as the test set.
- `window_size`: Specify the number of past days (D) to use for predicting the next day (D+1).


For a complete list of arguments, refer to the `parse_args` function in `utils.py`.

### Visualizing Results
The tool generates various plots to help understand the performance of the models. It includes:
- **Line plots**: Comparing the predicted and actual stock prices over time.
- **Residual plots**: Visualizing the residuals (differences between true and predicted values) over time to identify any patterns or anomalies.
- **Scatter plots**: Comparing the true prices with the predicted prices to assess the model's accuracy visually.

These visualizations help in analyzing the accuracy and effectiveness of the predictions.

## Model

## Results

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact m11207330@mail.ntust.edu.tw
