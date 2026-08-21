# Package index

## Double machine learning data class

- [`DoubleMLData`](DoubleMLData.md) : Double machine learning
  data-backend
- [`DoubleMLClusterData`](DoubleMLClusterData.md) : Double machine
  learning data-backend for data with cluster variables
- [`double_ml_data_from_data_frame()`](double_ml_data_from_data_frame.md)
  : Wrapper for Double machine learning data-backend initialization from
  data.frame.
- [`double_ml_data_from_matrix()`](double_ml_data_from_matrix.md) :
  Wrapper for Double machine learning data-backend initialization from
  matrix.

## Abstract base class for double machine learning models

- [`DoubleML`](DoubleML.md) : Abstract class DoubleML

## Double machine learning models

- [`DoubleMLPLR`](DoubleMLPLR.md) : Double machine learning for
  partially linear regression models
- [`DoubleMLPLIV`](DoubleMLPLIV.md) : Double machine learning for
  partially linear IV regression models
- [`DoubleMLIRM`](DoubleMLIRM.md) : Double machine learning for
  interactive regression models
- [`DoubleMLIIVM`](DoubleMLIIVM.md) : Double machine learning for
  interactive IV regression models
- [`DoubleMLSSM`](DoubleMLSSM.md) : Double machine learning for sample
  selection models

## Datasets module

- [`fetch_401k()`](fetch_401k.md) : Data set on financial wealth and
  401(k) plan participation.
- [`fetch_bonus()`](fetch_bonus.md) : Data set on the Pennsylvania
  Reemployment Bonus experiment.

## Datasets generators

- [`make_plr_CCDDHNR2018()`](make_plr_CCDDHNR2018.md) : Generates data
  from a partially linear regression model used in Chernozhukov et al.
  (2018)
- [`make_pliv_CHS2015()`](make_pliv_CHS2015.md) : Generates data from a
  partially linear IV regression model used in Chernozhukov, Hansen and
  Spindler (2015).
- [`make_irm_data()`](make_irm_data.md) : Generates data from a
  interactive regression (IRM) model.
- [`make_iivm_data()`](make_iivm_data.md) : Generates data from a
  interactive IV regression (IIVM) model.
- [`make_plr_turrell2018()`](make_plr_turrell2018.md) : Generates data
  from a partially linear regression model used in a blog article by
  Turrell (2018).
- [`make_pliv_multiway_cluster_CKMS2021()`](make_pliv_multiway_cluster_CKMS2021.md)
  : Generates data from a partially linear IV regression model with
  multiway cluster sample used in Chiang et al. (2021).
- [`make_ssm_data()`](make_ssm_data.md) : Generates data from a sample
  selection model (SSM).
