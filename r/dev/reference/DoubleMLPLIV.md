# Double machine learning for partially linear IV regression models

Double machine learning for partially linear IV regression models.

## Format

[R6::R6Class](https://r6.r-lib.org/reference/R6Class.html) object
inheriting from
[DoubleML](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.md).

## Details

Partially linear IV regression (PLIV) models take the form

\\Y - D\theta_0 = g_0(X) + \zeta\\,

\\Z = m_0(X) + V\\,

with \\E\[\zeta\|Z,X\]=0\\ and \\E\[V\|X\] = 0\\. \\Y\\ is the outcome
variable, \\D\\ is the policy variable of interest and \\Z\\ denotes one
or multiple instrumental variables. The high-dimensional vector \\X =
(X_1, \ldots, X_p)\\ consists of other confounding covariates, and
\\\zeta\\ and \\V\\ are stochastic errors.

## See also

Other DoubleML:
[`DoubleML`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.md),
[`DoubleMLIIVM`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLIIVM.md),
[`DoubleMLIRM`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLIRM.md),
[`DoubleMLPLR`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLPLR.md),
[`DoubleMLSSM`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLSSM.md)

## Super class

[`DoubleML`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.md)
-\> `DoubleMLPLIV`

## Active bindings

- `partialX`:

  (`logical(1)`)  
  Indicates whether covariates \\X\\ should be partialled out.

- `partialZ`:

  (`logical(1)`)  
  Indicates whether instruments \\Z\\ should be partialled out.

## Methods

### Public methods

- [`DoubleMLPLIV$new()`](#method-DoubleMLPLIV-initialize)

- [`DoubleMLPLIV$set_ml_nuisance_params()`](#method-DoubleMLPLIV-set_ml_nuisance_params)

- [`DoubleMLPLIV$tune()`](#method-DoubleMLPLIV-tune)

- [`DoubleMLPLIV$clone()`](#method-DoubleMLPLIV-clone)

Inherited methods

- [`DoubleML$bootstrap()`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.html#method-bootstrap)
- [`DoubleML$confint()`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.html#method-confint)
- [`DoubleML$fit()`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.html#method-fit)
- [`DoubleML$get_params()`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.html#method-get_params)
- [`DoubleML$learner_names()`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.html#method-learner_names)
- [`DoubleML$p_adjust()`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.html#method-p_adjust)
- [`DoubleML$params_names()`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.html#method-params_names)
- [`DoubleML$print()`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.html#method-print)
- [`DoubleML$set_sample_splitting()`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.html#method-set_sample_splitting)
- [`DoubleML$split_samples()`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.html#method-split_samples)
- [`DoubleML$summary()`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.html#method-summary)

------------------------------------------------------------------------

### `DoubleMLPLIV$new()`

Creates a new instance of this R6 class.

#### Usage

    DoubleMLPLIV$new(
      data,
      ml_l,
      ml_m,
      ml_r,
      ml_g = NULL,
      partialX = TRUE,
      partialZ = FALSE,
      n_folds = 5,
      n_rep = 1,
      score = "partialling out",
      dml_procedure = "dml2",
      draw_sample_splitting = TRUE,
      apply_cross_fitting = TRUE
    )

#### Arguments

- `data`:

  (`DoubleMLData`)  
  The `DoubleMLData` object providing the data and specifying the
  variables of the causal model.

- `ml_l`:

  ([`LearnerRegr`](https://mlr3.mlr-org.com/reference/LearnerRegr.html),
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html),
  `character(1)`)  
  A learner of the class
  [`LearnerRegr`](https://mlr3.mlr-org.com/reference/LearnerRegr.html),
  which is available from [mlr3](https://mlr3.mlr-org.com/index.html) or
  its extension packages
  [mlr3learners](https://mlr3learners.mlr-org.com/) or
  [mlr3extralearners](https://mlr3extralearners.mlr-org.com/).
  Alternatively, a
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) object
  with public field `task_type = "regr"` can be passed, for example of
  class
  [`GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html).
  The learner can possibly be passed with specified parameters, for
  example `lrn("regr.cv_glmnet", s = "lambda.min")`.  
  `ml_l` refers to the nuisance function \\l_0(X) = E\[Y\|X\]\\.

- `ml_m`:

  ([`LearnerRegr`](https://mlr3.mlr-org.com/reference/LearnerRegr.html),
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html),
  `character(1)`)  
  A learner of the class
  [`LearnerRegr`](https://mlr3.mlr-org.com/reference/LearnerRegr.html),
  which is available from [mlr3](https://mlr3.mlr-org.com/index.html) or
  its extension packages
  [mlr3learners](https://mlr3learners.mlr-org.com/) or
  [mlr3extralearners](https://mlr3extralearners.mlr-org.com/).
  Alternatively, a
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) object
  with public field `task_type = "regr"` can be passed, for example of
  class
  [`GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html).
  The learner can possibly be passed with specified parameters, for
  example `lrn("regr.cv_glmnet", s = "lambda.min")`.  
  `ml_m` refers to the nuisance function \\m_0(X) = E\[Z\|X\]\\.

- `ml_r`:

  ([`LearnerRegr`](https://mlr3.mlr-org.com/reference/LearnerRegr.html),
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html),
  `character(1)`)  
  A learner of the class
  [`LearnerRegr`](https://mlr3.mlr-org.com/reference/LearnerRegr.html),
  which is available from [mlr3](https://mlr3.mlr-org.com/index.html) or
  its extension packages
  [mlr3learners](https://mlr3learners.mlr-org.com/) or
  [mlr3extralearners](https://mlr3extralearners.mlr-org.com/).
  Alternatively, a
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) object
  with public field `task_type = "regr"` can be passed, for example of
  class
  [`GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html).
  The learner can possibly be passed with specified parameters, for
  example `lrn("regr.cv_glmnet", s = "lambda.min")`.  
  `ml_r` refers to the nuisance function \\r_0(X) = E\[D\|X\]\\.

- `ml_g`:

  ([`LearnerRegr`](https://mlr3.mlr-org.com/reference/LearnerRegr.html),
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html),
  `character(1)`)  
  A learner of the class
  [`LearnerRegr`](https://mlr3.mlr-org.com/reference/LearnerRegr.html),
  which is available from [mlr3](https://mlr3.mlr-org.com/index.html) or
  its extension packages
  [mlr3learners](https://mlr3learners.mlr-org.com/) or
  [mlr3extralearners](https://mlr3extralearners.mlr-org.com/).
  Alternatively, a
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) object
  with public field `task_type = "regr"` can be passed, for example of
  class
  [`GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html).
  The learner can possibly be passed with specified parameters, for
  example `lrn("regr.cv_glmnet", s = "lambda.min")`.  
  `ml_g` refers to the nuisance function \\g_0(X) = E\[Y -
  D\theta_0\|X\]\\. Note: The learner `ml_g` is only required for the
  score `'IV-type'`. Optionally, it can be specified and estimated for
  callable scores.

- `partialX`:

  (`logical(1)`)  
  Indicates whether covariates \\X\\ should be partialled out. Default
  is `TRUE`.

- `partialZ`:

  (`logical(1)`)  
  Indicates whether instruments \\Z\\ should be partialled out. Default
  is `FALSE`.

- `n_folds`:

  (`integer(1)`)  
  Number of folds. Default is `5`.

- `n_rep`:

  (`integer(1)`)  
  Number of repetitions for the sample splitting. Default is `1`.

- `score`:

  (`character(1)`, `function()`)  
  A `character(1)` (`"partialling out"` or `"IV-type"`) or a
  `function()` specifying the score function. If a `function()` is
  provided, it must be of the form
  `function(y, z, d, l_hat, m_hat, r_hat, g_hat, smpls)` and the
  returned output must be a named
  [`list()`](https://rdrr.io/r/base/list.html) with elements `psi_a` and
  `psi_b`. Default is `"partialling out"`.

- `dml_procedure`:

  (`character(1)`)  
  A `character(1)` (`"dml1"` or `"dml2"`) specifying the double machine
  learning algorithm. Default is `"dml2"`.

- `draw_sample_splitting`:

  (`logical(1)`)  
  Indicates whether the sample splitting should be drawn during
  initialization of the object. Default is `TRUE`.

- `apply_cross_fitting`:

  (`logical(1)`)  
  Indicates whether cross-fitting should be applied. Default is `TRUE`.

------------------------------------------------------------------------

### `DoubleMLPLIV$set_ml_nuisance_params()`

Set hyperparameters for the nuisance models of DoubleML models.

Note that in the current implementation, either all parameters have to
be set globally or all parameters have to be provided fold-specific.

#### Usage

    DoubleMLPLIV$set_ml_nuisance_params(
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

### `DoubleMLPLIV$tune()`

Hyperparameter-tuning for DoubleML models.

The hyperparameter-tuning is performed using the tuning methods provided
in the [mlr3tuning](https://mlr3tuning.mlr-org.com/) package. For more
information on tuning in [mlr3](https://mlr3.mlr-org.com/), we refer to
the section on parameter tuning in the [mlr3
book](https://mlr3book.mlr-org.com/chapters/chapter4/hyperparameter_optimization.html).

#### Usage

    DoubleMLPLIV$tune(
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

### `DoubleMLPLIV$clone()`

The objects of this class are cloneable with this method.

#### Usage

    DoubleMLPLIV$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# \donttest{
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(data.table)
set.seed(2)
ml_l = lrn("regr.ranger", num.trees = 100, mtry = 20, min.node.size = 2, max.depth = 5)
ml_m = ml_l$clone()
ml_r = ml_l$clone()
obj_dml_data = make_pliv_CHS2015(alpha = 1, n_obs = 500, dim_x = 20, dim_z = 1)
dml_pliv_obj = DoubleMLPLIV$new(obj_dml_data, ml_l, ml_m, ml_r)
dml_pliv_obj$fit()
dml_pliv_obj$summary()
#> Estimates and significance testing of the effect of target variables
#>   Estimate. Std. Error t value Pr(>|t|)    
#> d    0.9708     0.1037   9.363   <2e-16 ***
#> ---
#> Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#> 
#> 
# }

if (FALSE) { # \dontrun{
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(data.table)
set.seed(2)
ml_l = lrn("regr.rpart")
ml_m = ml_l$clone()
ml_r = ml_l$clone()
obj_dml_data = make_pliv_CHS2015(
  alpha = 1, n_obs = 500, dim_x = 20,
  dim_z = 1
)
dml_pliv_obj = DoubleMLPLIV$new(obj_dml_data, ml_l, ml_m, ml_r)
param_grid = list(
  "ml_l" = paradox::ps(
    cp = paradox::p_dbl(lower = 0.01, upper = 0.02),
    minsplit = paradox::p_int(lower = 1, upper = 2)
  ),
  "ml_m" = paradox::ps(
    cp = paradox::p_dbl(lower = 0.01, upper = 0.02),
    minsplit = paradox::p_int(lower = 1, upper = 2)
  ),
  "ml_r" = paradox::ps(
    cp = paradox::p_dbl(lower = 0.01, upper = 0.02),
    minsplit = paradox::p_int(lower = 1, upper = 2)
  )
)

# minimum requirements for tune_settings
tune_settings = list(
  terminator = mlr3tuning::trm("evals", n_evals = 5),
  algorithm = mlr3tuning::tnr("grid_search", resolution = 5)
)
dml_pliv_obj$tune(param_set = param_grid, tune_settings = tune_settings)
dml_pliv_obj$fit()
dml_pliv_obj$summary()
} # }
```
