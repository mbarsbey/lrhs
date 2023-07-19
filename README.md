# Source Code for the Paper: _Modeling Hierarchical Seasonality through Low-Rank Tensor Decompositions in Time Series Analysis_

This repository contains the source code that accompanies the paper _Modeling Hierarchical Seasonality through Low-Rank Tensor Decompositions in Time Series Analysis_, published in _IEEE Access_, and authored by Melih Barsbey and Taylan Cemgil.

The code mainly targets to replicate the results presented in Section 5 of the main paper or experiment with other settings and/or datasets. The four Jupyter notebooks in the root folder are intended to facilitate the user's access to the code and its functionalities. Each notebook includes more detailed information regarding the tasks it aims to accomplish. We provide a general overview here.

## Replicating the Univariate Experiments in Section 5.1

The experiments in Section 5.1 can be replicated using the Jupyter notebook `01_univariate_experiments.ipynb`. The user will have to download the datasets to be used in this experiment from the link provided within the notebook as the datasets are by themselves too large to store in this repository directly.

## Replicating the NYC Yellow Taxi Experiments in Section 5.2

The experiments in Section 5.2 can be replicated using the Jupyter notebook `02_nyc_yt_experiments.ipynb`. The trained model and a subset of the original data has been provided within this repository. The user can download the full dataset through the link provided within the notebook and use the repository to experiment with it, if desired.

## Replicating the BART Ridership Experiments in Section 5.3

The experiments in Section 5.3 can be replicated using the Jupyter notebook `03_bart_ridership_experiments.ipynb`. Here too, the trained model and a subset of the original data has been provided. Like above, a link has been provided within the notebook to download the full dataset, should the reader want to analyze it with our framework.

## Replicating the Imputation Experiments in Section 5.4

The imputation experiments in Section 5.4 can be replicated with the Jupyter notebook `04_imputation_experiments.ipynb`. Here, the notebook provides code to automatically download the data from Chen et al.'s (2019) public repository. After the data has been obtained, the experiments can be conducted according to the hyperparameters in the main paper, or new ones should the reader want to explore.

Chen, X., He, Z., & Sun, L. (2019). A Bayesian tensor decomposition approach for spatiotemporal traffic data imputation. Transportation Research Part C: Emerging Technologies, 98, 73–84.

## License

This project is licensed under the terms of the MIT license. For more information, please see [LICENSE](/LICENSE) in this repository.

For univariate experiments, this project modifies and uses code from another open source project by A. Caner Türkmen and Melih Barsbey, also under the MIT license. Please see [LICENSE](/lrhs/univariate/LICENSE) for more details.