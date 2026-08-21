# Wrapper for Double machine learning data-backend initialization from matrix.

Initalization of DoubleMLData from
[`matrix()`](https://rdrr.io/r/base/matrix.html) objects.

## Usage

``` r
double_ml_data_from_matrix(
  X = NULL,
  y,
  d,
  z = NULL,
  s = NULL,
  cluster_vars = NULL,
  data_class = "DoubleMLData",
  use_other_treat_as_covariate = TRUE
)
```

## Arguments

- X:

  ([`matrix()`](https://rdrr.io/r/base/matrix.html))  
  Matrix of covariates.

- y:

  ([`numeric()`](https://rdrr.io/r/base/numeric.html))  
  Vector of outcome variable.

- d:

  ([`matrix()`](https://rdrr.io/r/base/matrix.html))  
  Matrix of treatment variables.

- z:

  ([`matrix()`](https://rdrr.io/r/base/matrix.html))  
  Matrix of instruments.

- s:

  ([`numeric()`](https://rdrr.io/r/base/numeric.html))  
  Vector of the score or selection variable (only relevant for SSM
  models).

- cluster_vars:

  ([`matrix()`](https://rdrr.io/r/base/matrix.html))  
  Matrix of cluster variables.

- data_class:

  (`character(1)`)  
  Class of returned object. By default, an object of class
  `DoubleMLData` is returned. Setting `data_class = "data.table"`
  returns an object of class `data.table`.

- use_other_treat_as_covariate:

  (`logical(1)`)  
  Indicates whether in the multiple-treatment case the other treatment
  variables should be added as covariates. Default is `TRUE`.

## Value

Creates a new instance of class `DoubleMLData`.

## Examples

``` r
matrix_list = make_plr_CCDDHNR2018(return_type = "matrix")
obj_dml_data = double_ml_data_from_matrix(
  X = matrix_list$X,
  y = matrix_list$y,
  d = matrix_list$d)
```
