# Product Success Prediction Model Selection

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)

This project compares different machine learning models to predict the success of fashion products using historical data.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to predict the success of fashion products using machine learning techniques. It compares the performance of three different models: Random Forest, XGBoost, and Artificial Neural Network (ANN). The project includes data preprocessing, model training with hyperparameter tuning, and performance evaluation.

## Features

- Data preprocessing and feature engineering
- Implementation of Random Forest, XGBoost, and ANN models
- Hyperparameter tuning for each model
- Model performance comparison
- Feature importance analysis

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/product-success-prediction.git
   cd product-success-prediction
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have the `historic.csv` file in the `data/` directory.

2. Run the Jupyter notebook:
   ```
   jupyter notebook
   ```

3. Open `model_selection.ipynb` and run all cells.

4. The notebook will generate results and visualizations for model comparison.

## Models

The project compares the following models:

1. **Random Forest**: An ensemble learning method using multiple decision trees.
2. **XGBoost**: A gradient boosting algorithm known for its performance and speed.
3. **Artificial Neural Network (ANN)**: A multi-layer perceptron with dense layers and dropout.

Each model undergoes hyperparameter tuning to optimize performance.

## Results

The notebook generates:

- Accuracy scores for each model
- Classification reports (precision, recall, F1-score)
- Feature importance visualizations
- Model comparison plots

Refer to the notebook for detailed results and interpretations.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

Project Link: [https://github.com/your-username/product-success-prediction](https://github.com/your-username/product-success-prediction)
