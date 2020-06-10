# Installation
Install dependencies
```
make install-configure-poetry
```

Run tests
```
make test
```

To see examples of running fitting and hyperparameters tuning refer to tests.

# Details
1. LightGBM is used as an estimator
2. Bayesian Optimization package is used for parameters tuning
3. Input data can have numeric and categorical variables. User can specify categorical columns otherwise they will be extracted automatically depending on dataframe column dtype. Is user specifies not all the categorical columns which exist in the dataset, categorical columns list will be automatically extended.
4. All the categorical values which were not observed while training are mapped to unseen category. 