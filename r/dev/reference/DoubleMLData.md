# Double machine learning data-backend

Double machine learning data-backend.

`DoubleMLData` objects can be initialized from a
[data.table](https://rdrr.io/pkg/data.table/man/data.table.html).
Alternatively `DoubleML` provides functions to initialize from a
collection of `matrix` objects or a `data.frame`. The following
functions can be used to create a new instance of `DoubleMLData`.

- `DoubleMLData$new()` for initialization from a `data.table`.

- [`double_ml_data_from_matrix()`](double_ml_data_from_matrix.md) for
  initialization from `matrix` objects,

- [`double_ml_data_from_data_frame()`](double_ml_data_from_data_frame.md)
  for initialization from a `data.frame`.

## Active bindings

- `all_variables`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  All variables available in the dataset.

- `d_cols`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The treatment variable(s).

- `data`:

  ([`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html))  
  Data object.

- `data_model`:

  ([`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html))  
  Internal data object that implements the causal model as specified by
  the user via `y_col`, `d_cols`, `x_cols` and `z_cols`.

- `n_instr`:

  (`NULL`, `integer(1)`)  
  The number of instruments.

- `n_obs`:

  (`integer(1)`)  
  The number of observations.

- `n_treat`:

  (`integer(1)`)  
  The number of treatment variables.

- `other_treat_cols`:

  (`NULL`, [`character()`](https://rdrr.io/r/base/character.html))  
  If `use_other_treat_as_covariate` is `TRUE`, `other_treat_cols` are
  the treatment variables that are not "active" in the
  multiple-treatment case. These variables then are internally added to
  the covariates `x_cols` during the fitting stage. If
  `use_other_treat_as_covariate` is `FALSE`, `other_treat_cols` is
  `NULL`.

- `treat_col`:

  (`character(1)`)  
  "Active" treatment variable in the multiple-treatment case.

- `use_other_treat_as_covariate`:

  (`logical(1)`)  
  Indicates whether in the multiple-treatment case the other treatment
  variables should be added as covariates. Default is `TRUE`.

- `x_cols`:

  (`NULL`, [`character()`](https://rdrr.io/r/base/character.html))  
  The covariates. If `NULL`, all variables (columns of `data`) which are
  neither specified as outcome variable `y_col`, nor as treatment
  variables `d_cols`, nor as instrumental variables `z_cols` are used as
  covariates. Default is `NULL`.

- `y_col`:

  (`character(1)`)  
  The outcome variable.

- `z_cols`:

  (`NULL`, [`character()`](https://rdrr.io/r/base/character.html))  
  The instrumental variables. Default is `NULL`.

- `s_col`:

  (`NULL`, [`character()`](https://rdrr.io/r/base/character.html))  
  The score or selection variable (only relevant/used for SSM
  Estimators). Default is `NULL`.

## Methods

### Public methods

- [`DoubleMLData$new()`](#method-DoubleMLData-new)

- [`DoubleMLData$print()`](#method-DoubleMLData-print)

- [`DoubleMLData$set_data_model()`](#method-DoubleMLData-set_data_model)

- [`DoubleMLData$clone()`](#method-DoubleMLData-clone)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    DoubleMLData$new(
      data = NULL,
      x_cols = NULL,
      y_col = NULL,
      d_cols = NULL,
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

Print DoubleMLData objects.

#### Usage

    DoubleMLData$print()

------------------------------------------------------------------------

### Method `set_data_model()`

Setter function for `data_model`. The function implements the causal
model as specified by the user via `y_col`, `d_cols`, `x_cols` and
`z_cols` and assigns the role for the treatment variables in the
multiple-treatment case.

#### Usage

    DoubleMLData$set_data_model(treatment_var)

#### Arguments

- `treatment_var`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Active treatment variable that will be set to `treat_col`.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    DoubleMLData$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
library(DoubleML)
df = make_plr_CCDDHNR2018(return_type = "data.table")
obj_dml_data = DoubleMLData$new(df,
  y_col = "y",
  d_cols = "d")
```
