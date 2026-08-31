# Abstract class DoubleML

Abstract base class that can't be initialized.

## Format

[R6::R6Class](https://r6.r-lib.org/reference/R6Class.html) object.

## See also

Other DoubleML:
[`DoubleMLIIVM`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLIIVM.md),
[`DoubleMLIRM`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLIRM.md),
[`DoubleMLPLIV`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLPLIV.md),
[`DoubleMLPLR`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLPLR.md),
[`DoubleMLSSM`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLSSM.md)

## Active bindings

- `all_coef`:

  ([`matrix()`](https://rdrr.io/r/base/matrix.html))  
  Estimates of the causal parameter(s) for the `n_rep` different sample
  splits after calling `fit()`.

- `all_dml1_coef`:

  ([`array()`](https://rdrr.io/r/base/array.html))  
  Estimates of the causal parameter(s) for the `n_rep` different sample
  splits after calling `fit()` with `dml_procedure = "dml1"`.

- `all_se`:

  ([`matrix()`](https://rdrr.io/r/base/matrix.html))  
  Standard errors of the causal parameter(s) for the `n_rep` different
  sample splits after calling `fit()`.

- `apply_cross_fitting`:

  (`logical(1)`)  
  Indicates whether cross-fitting should be applied. Default is `TRUE`.

- `boot_coef`:

  ([`matrix()`](https://rdrr.io/r/base/matrix.html))  
  Bootstrapped coefficients for the causal parameter(s) after calling
  `fit()` and `bootstrap()`.

- `boot_t_stat`:

  ([`matrix()`](https://rdrr.io/r/base/matrix.html))  
  Bootstrapped t-statistics for the causal parameter(s) after calling
  `fit()` and `bootstrap()`.

- `coef`:

  ([`numeric()`](https://rdrr.io/r/base/numeric.html))  
  Estimates for the causal parameter(s) after calling `fit()`.

- `data`:

  ([`data.table`](https://rdrr.io/pkg/data.table/man/data.table.html))  
  Data object.

- `dml_procedure`:

  (`character(1)`)  
  A [`character()`](https://rdrr.io/r/base/character.html) (`"dml1"` or
  `"dml2"`) specifying the double machine learning algorithm. Default is
  `"dml2"`.

- `draw_sample_splitting`:

  (`logical(1)`)  
  Indicates whether the sample splitting should be drawn during
  initialization of the object. Default is `TRUE`.

- `learner`:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  The machine learners for the nuisance functions.

- `n_folds`:

  (`integer(1)`)  
  Number of folds. Default is `5`.

- `n_rep`:

  (`integer(1)`)  
  Number of repetitions for the sample splitting. Default is `1`.

- `params`:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  The hyperparameters of the learners.

- `psi`:

  ([`array()`](https://rdrr.io/r/base/array.html))  
  Value of the score function \\\psi(W;\theta, \eta)=\psi_a(W;\eta)
  \theta + \psi_b (W; \eta)\\ after calling `fit()`.

- `psi_a`:

  ([`array()`](https://rdrr.io/r/base/array.html))  
  Value of the score function component \\\psi_a(W;\eta)\\ after calling
  `fit()`.

- `psi_b`:

  ([`array()`](https://rdrr.io/r/base/array.html))  
  Value of the score function component \\\psi_b(W;\eta)\\ after calling
  `fit()`.

- `predictions`:

  ([`array()`](https://rdrr.io/r/base/array.html))  
  Predictions of the nuisance models after calling
  `fit(store_predictions=TRUE)`.

- `models`:

  ([`array()`](https://rdrr.io/r/base/array.html))  
  The fitted nuisance models after calling `fit(store_models=TRUE)`.

- `pval`:

  ([`numeric()`](https://rdrr.io/r/base/numeric.html))  
  p-values for the causal parameter(s) after calling `fit()`.

- `score`:

  (`character(1)`, `function()`)  
  A `character(1)` or `function()` specifying the score function.

- `se`:

  ([`numeric()`](https://rdrr.io/r/base/numeric.html))  
  Standard errors for the causal parameter(s) after calling `fit()`.

- `smpls`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  The partition used for cross-fitting.

- `smpls_cluster`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  The partition of clusters used for cross-fitting.

- `t_stat`:

  ([`numeric()`](https://rdrr.io/r/base/numeric.html))  
  t-statistics for the causal parameter(s) after calling `fit()`.

- `tuning_res`:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  Results from hyperparameter tuning.

## Methods

### Public methods

- [`DoubleML$new()`](#method-DoubleML-initialize)

- [`DoubleML$print()`](#method-DoubleML-print)

- [`DoubleML$fit()`](#method-DoubleML-fit)

- [`DoubleML$bootstrap()`](#method-DoubleML-bootstrap)

- [`DoubleML$split_samples()`](#method-DoubleML-split_samples)

- [`DoubleML$set_sample_splitting()`](#method-DoubleML-set_sample_splitting)

- [`DoubleML$tune()`](#method-DoubleML-tune)

- [`DoubleML$summary()`](#method-DoubleML-summary)

- [`DoubleML$confint()`](#method-DoubleML-confint)

- [`DoubleML$learner_names()`](#method-DoubleML-learner_names)

- [`DoubleML$params_names()`](#method-DoubleML-params_names)

- [`DoubleML$set_ml_nuisance_params()`](#method-DoubleML-set_ml_nuisance_params)

- [`DoubleML$p_adjust()`](#method-DoubleML-p_adjust)

- [`DoubleML$get_params()`](#method-DoubleML-get_params)

- [`DoubleML$clone()`](#method-DoubleML-clone)

------------------------------------------------------------------------

### `DoubleML$new()`

DoubleML is an abstract class that can't be initialized.

#### Usage

    DoubleML$new()

------------------------------------------------------------------------

### `DoubleML$print()`

Print DoubleML objects.

#### Usage

    DoubleML$print()

------------------------------------------------------------------------

### `DoubleML$fit()`

Estimate DoubleML models.

#### Usage

    DoubleML$fit(store_predictions = FALSE, store_models = FALSE)

#### Arguments

- `store_predictions`:

  (`logical(1)`)  
  Indicates whether the predictions for the nuisance functions should be
  stored in field `predictions`. Default is `FALSE`.

- `store_models`:

  (`logical(1)`)  
  Indicates whether the fitted models for the nuisance functions should
  be stored in field `models` if you want to analyze the models or
  extract information like variable importance. Default is `FALSE`.

#### Returns

self

------------------------------------------------------------------------

### `DoubleML$bootstrap()`

Multiplier bootstrap for DoubleML models.

#### Usage

    DoubleML$bootstrap(method = "normal", n_rep_boot = 500)

#### Arguments

- `method`:

  (`character(1)`)  
  A `character(1)` (`"Bayes"`, `"normal"` or `"wild"`) specifying the
  multiplier bootstrap method.

- `n_rep_boot`:

  (`integer(1)`)  
  The number of bootstrap replications.

#### Returns

self

------------------------------------------------------------------------

### `DoubleML$split_samples()`

Draw sample splitting for DoubleML models.

The samples are drawn according to the attributes `n_folds`, `n_rep` and
`apply_cross_fitting`.

#### Usage

    DoubleML$split_samples()

#### Returns

self

------------------------------------------------------------------------

### `DoubleML$set_sample_splitting()`

Set the sample splitting for DoubleML models.

The attributes `n_folds` and `n_rep` are derived from the provided
partition.

#### Usage

    DoubleML$set_sample_splitting(smpls)

#### Arguments

- `smpls`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  A nested [`list()`](https://rdrr.io/r/base/list.html). The outer lists
  needs to provide an entry per repeated sample splitting (length of the
  list is set as `n_rep`). The inner list is a named
  [`list()`](https://rdrr.io/r/base/list.html) with names `train_ids`
  and `test_ids`. The entries in `train_ids` and `test_ids` must be
  partitions per fold (length of `train_ids` and `test_ids` is set as
  `n_folds`).

#### Returns

self

#### Examples

    library(DoubleML)
    library(mlr3)
    set.seed(2)
    obj_dml_data = make_plr_CCDDHNR2018(n_obs=10)
    dml_plr_obj = DoubleMLPLR$new(obj_dml_data,
                                  lrn("regr.rpart"), lrn("regr.rpart"))

    # simple sample splitting with two folds and without cross-fitting
    smpls = list(list(train_ids = list(c(1, 2, 3, 4, 5)),
                      test_ids = list(c(6, 7, 8, 9, 10))))
    dml_plr_obj$set_sample_splitting(smpls)

    # sample splitting with two folds and cross-fitting but no repeated cross-fitting
    smpls = list(list(train_ids = list(c(1, 2, 3, 4, 5), c(6, 7, 8, 9, 10)),
                      test_ids = list(c(6, 7, 8, 9, 10), c(1, 2, 3, 4, 5))))
    dml_plr_obj$set_sample_splitting(smpls)

    # sample splitting with two folds and repeated cross-fitting with n_rep = 2
    smpls = list(list(train_ids = list(c(1, 2, 3, 4, 5), c(6, 7, 8, 9, 10)),
                      test_ids = list(c(6, 7, 8, 9, 10), c(1, 2, 3, 4, 5))),
                 list(train_ids = list(c(1, 3, 5, 7, 9), c(2, 4, 6, 8, 10)),
                      test_ids = list(c(2, 4, 6, 8, 10), c(1, 3, 5, 7, 9))))
    dml_plr_obj$set_sample_splitting(smpls)

------------------------------------------------------------------------

### `DoubleML$tune()`

Hyperparameter-tuning for DoubleML models.

The hyperparameter-tuning is performed using the tuning methods provided
in the [mlr3tuning](https://mlr3tuning.mlr-org.com/) package. For more
information on tuning in [mlr3](https://mlr3.mlr-org.com/), we refer to
the section on parameter tuning in the [mlr3
book](https://mlr3book.mlr-org.com/chapters/chapter4/hyperparameter_optimization.html).

#### Usage

    DoubleML$tune(
      param_set,
      tune_settings = list(n_folds_tune = 5, rsmp_tune = mlr3::rsmp("cv", folds = 5), measure
        = NULL, terminator = mlr3tuning::trm("evals", n_evals = 20), algorithm =
        mlr3tuning::tnr("grid_search"), resolution = 5),
      tune_on_folds = FALSE
    )

#### Arguments

- `param_set`:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  A named `list` with a parameter grid for each nuisance model/learner
  (see method `learner_names()`). The parameter grid must be an object
  of class
  [ParamSet](https://paradox.mlr-org.com/reference/ParamSet.html).

- `tune_settings`:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  A named [`list()`](https://rdrr.io/r/base/list.html) with arguments
  passed to the hyperparameter-tuning with
  [mlr3tuning](https://mlr3tuning.mlr-org.com/) to set up
  [TuningInstance](https://mlr3tuning.mlr-org.com/reference/TuningInstanceBatchSingleCrit.html)
  objects. `tune_settings` has entries

  - `terminator`
    ([Terminator](https://bbotk.mlr-org.com/reference/Terminator.html))  
    A [Terminator](https://bbotk.mlr-org.com/reference/Terminator.html)
    object. Specification of `terminator` is required to perform tuning.

  - `algorithm`
    ([Tuner](https://mlr3tuning.mlr-org.com/reference/Tuner.html) or
    `character(1)`)  
    A [Tuner](https://mlr3tuning.mlr-org.com/reference/Tuner.html)
    object (recommended) or key passed to the respective dictionary to
    specify the tuning algorithm used in
    [tnr()](https://mlr3tuning.mlr-org.com/reference/tnr.html).
    `algorithm` is passed as an argument to
    [tnr()](https://mlr3tuning.mlr-org.com/reference/tnr.html). If
    `algorithm` is not specified by the users, default is set to
    `"grid_search"`. If set to `"grid_search"`, then additional argument
    `"resolution"` is required.

  - `rsmp_tune`
    ([Resampling](https://mlr3.mlr-org.com/reference/Resampling.html) or
    `character(1)`)  
    A [Resampling](https://mlr3.mlr-org.com/reference/Resampling.html)
    object (recommended) or option passed to
    [rsmp()](https://mlr3.mlr-org.com/reference/mlr_sugar.html) to
    initialize a
    [Resampling](https://mlr3.mlr-org.com/reference/Resampling.html) for
    parameter tuning in `mlr3`. If not specified by the user, default is
    set to `"cv"` (cross-validation).

  - `n_folds_tune` (`integer(1)`, optional)  
    If `rsmp_tune = "cv"`, number of folds used for cross-validation. If
    not specified by the user, default is set to `5`.

  - `measure` (`NULL`, named
    [`list()`](https://rdrr.io/r/base/list.html), optional)  
    Named list containing the measures used for parameter tuning.
    Entries in list must either be
    [Measure](https://mlr3.mlr-org.com/reference/Measure.html) objects
    or keys to be passed to passed to
    [msr()](https://mlr3.mlr-org.com/reference/mlr_sugar.html). The
    names of the entries must match the learner names (see method
    `learner_names()`). If set to `NULL`, default measures are used,
    i.e., `"regr.mse"` for continuous outcome variables and
    `"classif.ce"` for binary outcomes.

  - `resolution` (`character(1)`)  
    The key passed to the respective dictionary to specify the tuning
    algorithm used in
    [tnr()](https://mlr3tuning.mlr-org.com/reference/tnr.html).
    `resolution` is passed as an argument to
    [tnr()](https://mlr3tuning.mlr-org.com/reference/tnr.html).

- `tune_on_folds`:

  (`logical(1)`)  
  Indicates whether the tuning should be done fold-specific or globally.
  Default is `FALSE`.

#### Returns

self

------------------------------------------------------------------------

### `DoubleML$summary()`

Summary for DoubleML models after calling `fit()`.

#### Usage

    DoubleML$summary(digits = max(3L, getOption("digits") - 3L))

#### Arguments

- `digits`:

  (`integer(1)`)  
  The number of significant digits to use when printing.

------------------------------------------------------------------------

### `DoubleML$confint()`

Confidence intervals for DoubleML models.

#### Usage

    DoubleML$confint(parm, joint = FALSE, level = 0.95)

#### Arguments

- `parm`:

  ([`numeric()`](https://rdrr.io/r/base/numeric.html) or
  [`character()`](https://rdrr.io/r/base/character.html))  
  A specification of which parameters are to be given confidence
  intervals among the variables for which inference was done, either a
  vector of numbers or a vector of names. If missing, all parameters are
  considered (default).

- `joint`:

  (`logical(1)`)  
  Indicates whether joint confidence intervals are computed. Default is
  `FALSE`.

- `level`:

  (`numeric(1)`)  
  The confidence level. Default is `0.95`.

#### Returns

A [`matrix()`](https://rdrr.io/r/base/matrix.html) with the confidence
interval(s).

------------------------------------------------------------------------

### `DoubleML$learner_names()`

Returns the names of the learners.

#### Usage

    DoubleML$learner_names()

#### Returns

[`character()`](https://rdrr.io/r/base/character.html) with names of
learners.

------------------------------------------------------------------------

### `DoubleML$params_names()`

Returns the names of the nuisance models with hyperparameters.

#### Usage

    DoubleML$params_names()

#### Returns

[`character()`](https://rdrr.io/r/base/character.html) with names of
nuisance models with hyperparameters.

------------------------------------------------------------------------

### `DoubleML$set_ml_nuisance_params()`

Set hyperparameters for the nuisance models of DoubleML models.

Note that in the current implementation, either all parameters have to
be set globally or all parameters have to be provided fold-specific.

#### Usage

    DoubleML$set_ml_nuisance_params(
      learner = NULL,
      treat_var = NULL,
      params,
      set_fold_specific = FALSE
    )

#### Arguments

- `learner`:

  (`character(1)`)  
  The nuisance model/learner (see method `params_names`).

- `treat_var`:

  (`character(1)`)  
  The treatment varaible (hyperparameters can be set treatment-variable
  specific).

- `params`:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  A named [`list()`](https://rdrr.io/r/base/list.html) with estimator
  parameters. Parameters are used for all folds by default.
  Alternatively, parameters can be passed in a fold-specific way if
  option `fold_specific`is `TRUE`. In this case, the outer list needs to
  be of length `n_rep` and the inner list of length `n_folds`.

- `set_fold_specific`:

  (`logical(1)`)  
  Indicates if the parameters passed in `params` should be passed in
  fold-specific way. Default is `FALSE`. If `TRUE`, the outer list needs
  to be of length `n_rep` and the inner list of length `n_folds`. Note
  that in the current implementation, either all parameters have to be
  set globally or all parameters have to be provided fold-specific.

#### Returns

self

------------------------------------------------------------------------

### `DoubleML$p_adjust()`

Multiple testing adjustment for DoubleML models.

#### Usage

    DoubleML$p_adjust(method = "romano-wolf", return_matrix = TRUE)

#### Arguments

- `method`:

  (`character(1)`)  
  A `character(1)`(`"romano-wolf"`, `"bonferroni"`, `"holm"`, etc)
  specifying the adjustment method. In addition to `"romano-wolf"`, all
  methods implemented in
  [p.adjust()](https://rdrr.io/r/stats/p.adjust.html) can be applied.
  Default is `"romano-wolf"`.

- `return_matrix`:

  (`logical(1)`)  
  Indicates if the output is returned as a matrix with corresponding
  coefficient names.

#### Returns

[`numeric()`](https://rdrr.io/r/base/numeric.html) with adjusted
p-values. If `return_matrix = TRUE`, a
[`matrix()`](https://rdrr.io/r/base/matrix.html) with adjusted p_values.

------------------------------------------------------------------------

### `DoubleML$get_params()`

Get hyperparameters for the nuisance model of DoubleML models.

#### Usage

    DoubleML$get_params(learner)

#### Arguments

- `learner`:

  (`character(1)`)  
  The nuisance model/learner (see method `params_names()`)

#### Returns

named [`list()`](https://rdrr.io/r/base/list.html)with paramers for the
nuisance model/learner.

------------------------------------------------------------------------

### `DoubleML$clone()`

The objects of this class are cloneable with this method.

#### Usage

    DoubleML$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r

## ------------------------------------------------
## Method `DoubleML$set_sample_splitting()`
## ------------------------------------------------

library(DoubleML)
library(mlr3)
set.seed(2)
obj_dml_data = make_plr_CCDDHNR2018(n_obs=10)
dml_plr_obj = DoubleMLPLR$new(obj_dml_data,
                              lrn("regr.rpart"), lrn("regr.rpart"))

# simple sample splitting with two folds and without cross-fitting
smpls = list(list(train_ids = list(c(1, 2, 3, 4, 5)),
                  test_ids = list(c(6, 7, 8, 9, 10))))
dml_plr_obj$set_sample_splitting(smpls)

# sample splitting with two folds and cross-fitting but no repeated cross-fitting
smpls = list(list(train_ids = list(c(1, 2, 3, 4, 5), c(6, 7, 8, 9, 10)),
                  test_ids = list(c(6, 7, 8, 9, 10), c(1, 2, 3, 4, 5))))
dml_plr_obj$set_sample_splitting(smpls)

# sample splitting with two folds and repeated cross-fitting with n_rep = 2
smpls = list(list(train_ids = list(c(1, 2, 3, 4, 5), c(6, 7, 8, 9, 10)),
                  test_ids = list(c(6, 7, 8, 9, 10), c(1, 2, 3, 4, 5))),
             list(train_ids = list(c(1, 3, 5, 7, 9), c(2, 4, 6, 8, 10)),
                  test_ids = list(c(2, 4, 6, 8, 10), c(1, 3, 5, 7, 9))))
dml_plr_obj$set_sample_splitting(smpls)
```
