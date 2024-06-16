# Stock-Price-Prediction

This repository contains code and resources for predicting stock prices using machine learning techniques. The goal is to develop a model that can accurately predict future stock prices based on historical data.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
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

## Usage

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
