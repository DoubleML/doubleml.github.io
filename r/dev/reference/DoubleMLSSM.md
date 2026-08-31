# Double machine learning for sample selection models

Double machine learning for sample selection models. Binary or multiple
treatment effect evaluation with double machine learning under sample
selection/outcome attrition. Potential outcomes \\Y(0)\\ and \\Y(1)\\
are estimated and ATE is returned as \\E\[Y(1) - Y(0)\]\\.

## Format

[R6::R6Class](https://r6.r-lib.org/reference/R6Class.html) object
inheriting from
[DoubleML](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.md).

## See also

Other DoubleML:
[`DoubleML`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.md),
[`DoubleMLIIVM`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLIIVM.md),
[`DoubleMLIRM`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLIRM.md),
[`DoubleMLPLIV`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLPLIV.md),
[`DoubleMLPLR`](https://docs.doubleml.org/r/stable/dev/reference/DoubleMLPLR.md)

## Super class

[`DoubleML`](https://docs.doubleml.org/r/stable/dev/reference/DoubleML.md)
-\> `DoubleMLSSM`

## Active bindings

- `trimming_rule`:

  (`character(1)`)  
  A `character(1)` specifying the trimming approach.

- `trimming_threshold`:

  (`numeric(1)`)  
  The threshold used for timming.

## Methods

### Public methods

- [`DoubleMLSSM$new()`](#method-DoubleMLSSM-initialize)

- [`DoubleMLSSM$set_ml_nuisance_params()`](#method-DoubleMLSSM-set_ml_nuisance_params)

- [`DoubleMLSSM$tune()`](#method-DoubleMLSSM-tune)

- [`DoubleMLSSM$clone()`](#method-DoubleMLSSM-clone)

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

### `DoubleMLSSM$new()`

Creates a new instance of this R6 class.

#### Usage

    DoubleMLSSM$new(
      data,
      ml_g,
      ml_pi,
      ml_m,
      n_folds = 5,
      n_rep = 1,
      score = "missing-at-random",
      normalize_ipw = FALSE,
      trimming_rule = "truncate",
      trimming_threshold = 1e-12,
      dml_procedure = "dml2",
      draw_sample_splitting = TRUE,
      apply_cross_fitting = TRUE
    )

#### Arguments

- `data`:

  (`DoubleMLData`)  
  The `DoubleMLData` object providing the data and specifying the
  variables of the causal model.

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
  `ml_g` refers to the nuisance function \\g_0(S,D,X) = E\[Y\|S,D,X\]\\.

- `ml_pi`:

  ([`LearnerClassif`](https://mlr3.mlr-org.com/reference/LearnerClassif.html),
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html),
  `character(1)`)  
  A learner of the class
  [`LearnerClassif`](https://mlr3.mlr-org.com/reference/LearnerClassif.html),
  which is available from [mlr3](https://mlr3.mlr-org.com/index.html) or
  its extension packages
  [mlr3learners](https://mlr3learners.mlr-org.com/) or
  [mlr3extralearners](https://mlr3extralearners.mlr-org.com/).
  Alternatively, a
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) object
  with public field `task_type = "classif"` can be passed, for example
  of class
  [`GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html).
  The learner can possibly be passed with specified parameters, for
  example `lrn("classif.cv_glmnet", s = "lambda.min")`.  
  `ml_pi` refers to the nuisance function \\pi_0(D,X) =
  Pr\[S=1\|D,X\]\\.

- `ml_m`:

  ([`LearnerRegr`](https://mlr3.mlr-org.com/reference/LearnerRegr.html),
  [`LearnerClassif`](https://mlr3.mlr-org.com/reference/LearnerClassif.html),
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html),
  `character(1)`)  
  A learner of the class
  [`LearnerClassif`](https://mlr3.mlr-org.com/reference/LearnerClassif.html),
  which is available from [mlr3](https://mlr3.mlr-org.com/index.html) or
  its extension packages
  [mlr3learners](https://mlr3learners.mlr-org.com/) or
  [mlr3extralearners](https://mlr3extralearners.mlr-org.com/).
  Alternatively, a
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) object
  with public field `task_type = "classif"` can be passed, for example
  of class
  [`GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html).
  The learner can possibly be passed with specified parameters, for
  example `lrn("classif.cv_glmnet", s = "lambda.min")`.  
  `ml_m` refers to the nuisance function \\m_0(X) = Pr\[D=1\|X\]\\.

- `n_folds`:

  (`integer(1)`)  
  Number of folds. Default is `5`.

- `n_rep`:

  (`integer(1)`)  
  Number of repetitions for the sample splitting. Default is `1`.

- `score`:

  (`character(1)`, `function()`)  
  A `character(1)` (`"missing-at-random"` or `"nonignorable"`)
  specifying the score function. Default is `"missing-at-random"`.

- `normalize_ipw`:

  (`logical(1)`)  
  Indicates whether the inverse probability weights are normalized.
  Default is `FALSE`.

- `trimming_rule`:

  (`character(1)`)  
  A `character(1)` (`"truncate"` is the only choice) specifying the
  trimming approach. Default is `"truncate"`.

- `trimming_threshold`:

  (`numeric(1)`)  
  The threshold used for timming. Default is `1e-12`.

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

### `DoubleMLSSM$set_ml_nuisance_params()`

Set hyperparameters for the nuisance models of DoubleML models.

Note that in the current implementation, either all parameters have to
be set globally or all parameters have to be provided fold-specific.

#### Usage

    DoubleMLSSM$set_ml_nuisance_params(
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

### `DoubleMLSSM$tune()`

Hyperparameter-tuning for DoubleML models.

The hyperparameter-tuning is performed using the tuning methods provided
in the [mlr3tuning](https://mlr3tuning.mlr-org.com/) package. For more
information on tuning in [mlr3](https://mlr3.mlr-org.com/), we refer to
the section on parameter tuning in the [mlr3
book](https://mlr3book.mlr-org.com/chapters/chapter4/hyperparameter_optimization.html).

#### Usage

    DoubleMLSSM$tune(
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

### `DoubleMLSSM$clone()`

The objects of this class are cloneable with this method.

#### Usage

    DoubleMLSSM$clone(deep = FALSE)

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
ml_g = lrn("regr.ranger",
  num.trees = 100, mtry = 20,
  min.node.size = 2, max.depth = 5
)
ml_m = lrn("classif.ranger",
  num.trees = 100, mtry = 20,
  min.node.size = 2, max.depth = 5
)
ml_pi = lrn("classif.ranger",
  num.trees = 100, mtry = 20,
  min.node.size = 2, max.depth = 5
)

n_obs = 2000
df = make_ssm_data(n_obs = n_obs, mar = TRUE, return_type = "data.table")
dml_data = DoubleMLData$new(df, y_col = "y", d_cols = "d", s_col = "s")
dml_ssm = DoubleMLSSM$new(dml_data, ml_g, ml_m, ml_pi, score = "missing-at-random")
dml_ssm$fit()
print(dml_ssm)
#> ================= DoubleMLSSM Object ==================
#> 
#> 
#> ------------------ Data summary      ------------------
#> Outcome variable: y
#> Treatment variable(s): d
#> Covariates: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, X31, X32, X33, X34, X35, X36, X37, X38, X39, X40, X41, X42, X43, X44, X45, X46, X47, X48, X49, X50, X51, X52, X53, X54, X55, X56, X57, X58, X59, X60, X61, X62, X63, X64, X65, X66, X67, X68, X69, X70, X71, X72, X73, X74, X75, X76, X77, X78, X79, X80, X81, X82, X83, X84, X85, X86, X87, X88, X89, X90, X91, X92, X93, X94, X95, X96, X97, X98, X99, X100
#> Instrument(s): 
#> Selection variable: s
#> No. Observations: 2000
#> 
#> ------------------ Score & algorithm ------------------
#> Score function: missing-at-random
#> DML algorithm: dml2
#> 
#> ------------------ Machine learner   ------------------
#> ml_g: regr.ranger
#> ml_pi: classif.ranger
#> ml_m: classif.ranger
#> 
#> ------------------ Resampling        ------------------
#> No. folds: 5
#> No. repeated sample splits: 1
#> Apply cross-fitting: TRUE
#> 
#> ------------------ Fit summary       ------------------
#>  Estimates and significance testing of the effect of target variables
#>   Estimate. Std. Error t value Pr(>|t|)    
#> d   1.06933    0.05862   18.24   <2e-16 ***
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
ml_g = lrn("regr.rpart")
ml_m = lrn("classif.rpart")
ml_pi = lrn("classif.rpart")
dml_data = make_ssm_data(n_obs = n_obs, mar = TRUE)
dml_ssm = DoubleMLSSM$new(dml_data,
  ml_g = ml_g, ml_m = ml_m, ml_pi = ml_pi,
  score = "missing-at-random"
)

param_grid = list(
  "ml_g" = paradox::ps(
    cp = paradox::p_dbl(lower = 0.01, upper = 0.02),
    minsplit = paradox::p_int(lower = 1, upper = 2)
  ),
  "ml_m" = paradox::ps(
    cp = paradox::p_dbl(lower = 0.01, upper = 0.02),
    minsplit = paradox::p_int(lower = 1, upper = 2)
  ),
  "ml_pi" = paradox::ps(
    cp = paradox::p_dbl(lower = 0.01, upper = 0.02),
    minsplit = paradox::p_int(lower = 1, upper = 2)
  )
)

# minimum requirements for tune_settings
tune_settings = list(
  terminator = mlr3tuning::trm("evals", n_evals = 5),
  algorithm = mlr3tuning::tnr("grid_search", resolution = 5)
)

dml_ssm$tune(param_set = param_grid, tune_settings = tune_settings)
dml_ssm$fit()
dml_ssm$summary()
} # }
```
