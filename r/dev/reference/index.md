# Package index

## Double machine learning data class

- [`DoubleMLData`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLData.md)
  : Double machine learning data-backend
- [`DoubleMLClusterData`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLClusterData.md)
  : Double machine learning data-backend for data with cluster variables
- [`double_ml_data_from_data_frame()`](https://docs.doubleml.org/r/stable/dev/reference/double_ml_data_from_data_frame.md)
  : Wrapper for Double machine learning data-backend initialization from
  data.frame.
- [`double_ml_data_from_matrix()`](https://docs.doubleml.org/r/stable/dev/reference/double_ml_data_from_matrix.md)
  : Wrapper for Double machine learning data-backend initialization from
  matrix.

## Abstract base class for double machine learning models

- [`DoubleML`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.md)
  : Abstract class DoubleML

## Double machine learning models

- [`DoubleMLPLR`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLPLR.md)
  : Double machine learning for partially linear regression models
- [`DoubleMLPLIV`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLPLIV.md)
  : Double machine learning for partially linear IV regression models
- [`DoubleMLIRM`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLIRM.md)
  : Double machine learning for interactive regression models
- [`DoubleMLIIVM`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLIIVM.md)
  : Double machine learning for interactive IV regression models
- [`DoubleMLSSM`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLSSM.md)
  : Double machine learning for sample selection models

## Datasets module

- [`fetch_401k()`](https://docs.doubleml.org/r/stable/dev/reference/fetch_401k.md)
  : Data set on financial wealth and 401(k) plan participation.
- [`fetch_bonus()`](https://docs.doubleml.org/r/stable/dev/reference/fetch_bonus.md)
  : Data set on the Pennsylvania Reemployment Bonus experiment.

## Datasets generators

- [`make_plr_CCDDHNR2018()`](https://docs.doubleml.org/r/stable/dev/reference/make_plr_CCDDHNR2018.md)
  : Generates data from a partially linear regression model used in
  Chernozhukov et al. (2018)
- [`make_pliv_CHS2015()`](https://docs.doubleml.org/r/stable/dev/reference/make_pliv_CHS2015.md)
  : Generates data from a partially linear IV regression model used in
  Chernozhukov, Hansen and Spindler (2015).
- [`make_irm_data()`](https://docs.doubleml.org/r/stable/dev/reference/make_irm_data.md)
  : Generates data from a interactive regression (IRM) model.
- [`make_iivm_data()`](https://docs.doubleml.org/r/stable/dev/reference/make_iivm_data.md)
  : Generates data from a interactive IV regression (IIVM) model.
- [`make_plr_turrell2018()`](https://docs.doubleml.org/r/stable/dev/reference/make_plr_turrell2018.md)
  : Generates data from a partially linear regression model used in a
  blog article by Turrell (2018).
- [`make_pliv_multiway_cluster_CKMS2021()`](https://docs.doubleml.org/r/stable/dev/reference/make_pliv_multiway_cluster_CKMS2021.md)
  : Generates data from a partially linear IV regression model with
  multiway cluster sample used in Chiang et al. (2021).
- [`make_ssm_data()`](https://docs.doubleml.org/r/stable/dev/reference/make_ssm_data.md)
  : Generates data from a sample selection model (SSM).
