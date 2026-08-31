# Wrapper for Double machine learning data-backend initialization from data.frame.

Initalization of DoubleMLData from `data.frame`.

## Usage

``` r
double_ml_data_from_data_frame(
  df,
  x_cols = NULL,
  y_col = NULL,
  d_cols = NULL,
  z_cols = NULL,
  s_col = NULL,
  cluster_cols = NULL,
  use_other_treat_as_covariate = TRUE
)
```

## Arguments

- df:

  ([`data.frame()`](https://rdrr.io/r/base/data.frame.html))  
  Data object.

- x_cols:

  (`NULL`, [`character()`](https://rdrr.io/r/base/character.html))  
  The covariates. If `NULL`, all variables (columns of `data`) which are
  neither specified as outcome variable `y_col`, nor as treatment
  variables `d_cols`, nor as instrumental variables `z_cols` are used as
  covariates. Default is `NULL`.

- y_col:

  (`character(1)`)  
  The outcome variable.

- d_cols:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The treatment variable(s).

- z_cols:

  (`NULL`, [`character()`](https://rdrr.io/r/base/character.html))  
  The instrumental variables. Default is `NULL`.

- s_col:

  (`NULL`, [`character()`](https://rdrr.io/r/base/character.html))  
  The score or selection variable (only relevant/used for SSM
  Estimators). Default is `NULL`.

- cluster_cols:

  (`NULL`, [`character()`](https://rdrr.io/r/base/character.html))  
  The cluster variables. Default is `NULL`.

- use_other_treat_as_covariate:

  (`logical(1)`)  
  Indicates whether in the multiple-treatment case the other treatment
  variables should be added as covariates. Default is `TRUE`.

## Value

Creates a new instance of class `DoubleMLData`.

## Examples

``` r
df = make_plr_CCDDHNR2018(return_type = "data.frame")
x_names = names(df)[grepl("X", names(df))]
obj_dml_data = double_ml_data_from_data_frame(
  df = df, x_cols = x_names,
  y_col = "y", d_cols = "d"
)
# Input: Data frame, Output: DoubleMLData object
```
