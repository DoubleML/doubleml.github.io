# Double machine learning data-backend for data with cluster variables

Double machine learning data-backend for data with cluster variables.

`DoubleMLClusterData` objects can be initialized from a
[data.table](https://rdrr.io/pkg/data.table/man/data.table.html).
Alternatively `DoubleML` provides functions to initialize from a
collection of `matrix` objects or a `data.frame`. The following
functions can be used to create a new instance of `DoubleMLClusterData`.

- `DoubleMLClusterData$new()` for initialization from a `data.table`.

- [`double_ml_data_from_matrix()`](double_ml_data_from_matrix.md) for
  initialization from `matrix` objects,

- [`double_ml_data_from_data_frame()`](double_ml_data_from_data_frame.md)
  for initialization from a `data.frame`.

## Super class

[`DoubleML::DoubleMLData`](DoubleMLData.md) -\> `DoubleMLClusterData`

## Active bindings

- `cluster_cols`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The cluster variable(s).

- `x_cols`:

  (`NULL`, [`character()`](https://rdrr.io/r/base/character.html))  
  The covariates. If `NULL`, all variables (columns of `data`) which are
  neither specified as outcome variable `y_col`, nor as treatment
  variables `d_cols`, nor as instrumental variables `z_cols`, nor as
  cluster variables `cluster_cols` are used as covariates. Default is
  `NULL`.

- `n_cluster_vars`:

  (`integer(1)`)  
  The number of cluster variables.

## Methods

### Public methods

- [`DoubleMLClusterData$new()`](#method-DoubleMLClusterData-new)

- [`DoubleMLClusterData$print()`](#method-DoubleMLClusterData-print)

- [`DoubleMLClusterData$set_data_model()`](#method-DoubleMLClusterData-set_data_model)

- [`DoubleMLClusterData$clone()`](#method-DoubleMLClusterData-clone)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    DoubleMLClusterData$new(
      data = NULL,
      x_cols = NULL,
      y_col = NULL,
      d_cols = NULL,
      cluster_cols = NULL,
      z_cols = NULL,
      s_col = NULL,
      use_other_treat_as_covariate = TRUE
    )

#### Arguments

- `data`:

  ([`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html),
  [`data.frame()`](https://rdrr.io/r/base/data.frame.html))  
  Data object.

- `x_cols`:

  (`NULL`, [`character()`](https://rdrr.io/r/base/character.html))  
  The covariates. If `NULL`, all variables (columns of `data`) which are
  neither specified as outcome variable `y_col`, nor as treatment
  variables `d_cols`, nor as instrumental variables `z_cols` are used as
  covariates. Default is `NULL`.

- `y_col`:

  (`character(1)`)  
  The outcome variable.

- `d_cols`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The treatment variable(s).

- `cluster_cols`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The cluster variable(s).

- `z_cols`:

  (`NULL`, [`character()`](https://rdrr.io/r/base/character.html))  
  The instrumental variables. Default is `NULL`.

- `s_col`:

  (`NULL`, [`character()`](https://rdrr.io/r/base/character.html))  
  The score or selection variable (only relevant/used for SSM
  Estimators). Default is `NULL`.

- `use_other_treat_as_covariate`:

  (`logical(1)`)  
  Indicates whether in the multiple-treatment case the other treatment
  variables should be added as covariates. Default is `TRUE`.

------------------------------------------------------------------------

### Method [`print()`](https://rdrr.io/r/base/print.html)

Print DoubleMLClusterData objects.

#### Usage

    DoubleMLClusterData$print()

------------------------------------------------------------------------

### Method `set_data_model()`

Setter function for `data_model`. The function implements the causal
model as specified by the user via `y_col`, `d_cols`, `x_cols`, `z_cols`
and `cluster_cols` and assigns the role for the treatment variables in
the multiple-treatment case.

#### Usage

    DoubleMLClusterData$set_data_model(treatment_var)

#### Arguments

- `treatment_var`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Active treatment variable that will be set to `treat_col`.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    DoubleMLClusterData$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
library(DoubleML)
dt = make_pliv_multiway_cluster_CKMS2021(return_type = "data.table")
obj_dml_data = DoubleMLClusterData$new(dt,
  y_col = "Y",
  d_cols = "D",
  z_cols = "Z",
  cluster_cols = c("cluster_var_i", "cluster_var_j"))
```
