# Double machine learning for interactive regression models

Double machine learning for interactive regression models.

## Format

[R6::R6Class](https://r6.r-lib.org/reference/R6Class.html) object
inheriting from [DoubleML](DoubleML.md).

## Details

Interactive regression (IRM) models take the form

\\Y = g_0(D,X) + U\\,

\\D = m_0(X) + V\\,

with \\E\[U\|X,D\]=0\\ and \\E\[V\|X\] = 0\\. \\Y\\ is the outcome
variable and \\D \in \\0,1\\\\ is the binary treatment variable. We
consider estimation of the average treamtent effects when treatment
effects are fully heterogeneous. Target parameters of interest in this
model are the average treatment effect (ATE),

\\\theta_0 = E\[g_0(1,X) - g_0(0,X)\]\\

and the average treament effect on the treated (ATTE),

\\\theta_0 = E\[g_0(1,X) - g_0(0,X)\|D=1\]\\.

## See also

Other DoubleML: [`DoubleML`](DoubleML.md),
[`DoubleMLIIVM`](DoubleMLIIVM.md), [`DoubleMLPLIV`](DoubleMLPLIV.md),
[`DoubleMLPLR`](DoubleMLPLR.md), [`DoubleMLSSM`](DoubleMLSSM.md)

## Super class

[`DoubleML::DoubleML`](DoubleML.md) -\> `DoubleMLIRM`

## Active bindings

- `trimming_rule`:

  (`character(1)`)  
  A `character(1)` specifying the trimming approach.

- `trimming_threshold`:

  (`numeric(1)`)  
  The threshold used for timming.

## Methods

### Public methods

- [`DoubleMLIRM$new()`](#method-DoubleMLIRM-new)

- [`DoubleMLIRM$clone()`](#method-DoubleMLIRM-clone)

Inherited methods

- [`DoubleML::DoubleML$bootstrap()`](DoubleML.html#method-bootstrap)
- [`DoubleML::DoubleML$confint()`](DoubleML.html#method-confint)
- [`DoubleML::DoubleML$fit()`](DoubleML.html#method-fit)
- [`DoubleML::DoubleML$get_params()`](DoubleML.html#method-get_params)
- [`DoubleML::DoubleML$learner_names()`](DoubleML.html#method-learner_names)
- [`DoubleML::DoubleML$p_adjust()`](DoubleML.html#method-p_adjust)
- [`DoubleML::DoubleML$params_names()`](DoubleML.html#method-params_names)
- [`DoubleML::DoubleML$print()`](DoubleML.html#method-print)
- [`DoubleML::DoubleML$set_ml_nuisance_params()`](DoubleML.html#method-set_ml_nuisance_params)
- [`DoubleML::DoubleML$set_sample_splitting()`](DoubleML.html#method-set_sample_splitting)
- [`DoubleML::DoubleML$split_samples()`](DoubleML.html#method-split_samples)
- [`DoubleML::DoubleML$summary()`](DoubleML.html#method-summary)
- [`DoubleML::DoubleML$tune()`](DoubleML.html#method-tune)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this R6 class.

#### Usage

    DoubleMLIRM$new(
      data,
      ml_g,
      ml_m,
      n_folds = 5,
      n_rep = 1,
      score = "ATE",
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
  [`LearnerClassif`](https://mlr3.mlr-org.com/reference/LearnerClassif.html),
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html),
  `character(1)`)  
  A learner of the class
  [`LearnerRegr`](https://mlr3.mlr-org.com/reference/LearnerRegr.html),
  which is available from [mlr3](https://mlr3.mlr-org.com/index.html) or
  its extension packages
  [mlr3learners](https://mlr3learners.mlr-org.com/) or
  [mlr3extralearners](https://mlr3extralearners.mlr-org.com/). For
  binary treatment outcomes, an object of the class
  [`LearnerClassif`](https://mlr3.mlr-org.com/reference/LearnerClassif.html)
  can be passed, for example
  `lrn("classif.cv_glmnet", s = "lambda.min")`. Alternatively, a
  [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) object
  with public field `task_type = "regr"` or `task_type = "classif"` can
  be passed, respectively, for example of class
  [`GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html).  
  `ml_g` refers to the nuisance function \\g_0(X) = E\[Y\|X,D\]\\.

- `ml_m`:

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
  `ml_m` refers to the nuisance function \\m_0(X) = E\[D\|X\]\\.

- `n_folds`:

  (`integer(1)`)  
  Number of folds. Default is `5`.

- `n_rep`:

  (`integer(1)`)  
  Number of repetitions for the sample splitting. Default is `1`.

- `score`:

  (`character(1)`, `function()`)  
  A `character(1)` (`"ATE"` or `ATTE`) or a `function()` specifying the
  score function. If a `function()` is provided, it must be of the form
  `function(y, d, g0_hat, g1_hat, m_hat, smpls)` and the returned output
  must be a named [`list()`](https://rdrr.io/r/base/list.html) with
  elements `psi_a` and `psi_b`. Default is `"ATE"`.

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

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    DoubleMLIRM$clone(deep = FALSE)

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
  min.node.size = 2, max.depth = 5)
ml_m = lrn("classif.ranger",
  num.trees = 100, mtry = 20,
  min.node.size = 2, max.depth = 5)
obj_dml_data = make_irm_data(theta = 0.5)
dml_irm_obj = DoubleMLIRM$new(obj_dml_data, ml_g, ml_m)
dml_irm_obj$fit()
dml_irm_obj$summary()
#> Estimates and significance testing of the effect of target variables
#>   Estimate. Std. Error t value Pr(>|t|)  
#> d    0.6658     0.2866   2.323   0.0202 *
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
obj_dml_data = make_irm_data(theta = 0.5)
dml_irm_obj = DoubleMLIRM$new(obj_dml_data, ml_g, ml_m)

param_grid = list(
  "ml_g" = paradox::ps(
    cp = paradox::p_dbl(lower = 0.01, upper = 0.02),
    minsplit = paradox::p_int(lower = 1, upper = 2)),
  "ml_m" = paradox::ps(
    cp = paradox::p_dbl(lower = 0.01, upper = 0.02),
    minsplit = paradox::p_int(lower = 1, upper = 2)))

# minimum requirements for tune_settings
tune_settings = list(
  terminator = mlr3tuning::trm("evals", n_evals = 5),
  algorithm = mlr3tuning::tnr("grid_search", resolution = 5))
dml_irm_obj$tune(param_set = param_grid, tune_settings = tune_settings)
dml_irm_obj$fit()
dml_irm_obj$summary()
} # }
```
