# Double machine learning for interactive IV regression models

Double machine learning for interactive IV regression models.

## Format

[R6::R6Class](https://r6.r-lib.org/reference/R6Class.html) object
inheriting from [DoubleML](DoubleML.md).

## Details

Interactive IV regression (IIVM) models take the form

\\Y = \ell_0(D,X) + \zeta\\,

\\Z = m_0(X) + V\\,

with \\E\[\zeta\|X,Z\]=0\\ and \\E\[V\|X\] = 0\\. \\Y\\ is the outcome
variable, \\D \in \\0,1\\\\ is the binary treatment variable and \\Z \in
\\0,1\\\\ is a binary instrumental variable. Consider the functions
\\g_0\\, \\r_0\\ and \\m_0\\, where \\g_0\\ maps the support of
\\(Z,X)\\ to \\R\\ and \\r_0\\ and \\m_0\\, respectively, map the
support of \\(Z,X)\\ and \\X\\ to \\(\epsilon, 1-\epsilon)\\ for some
\\\epsilon \in (1, 1/2)\\, such that

\\Y = g_0(Z,X) + \nu,\\

\\D = r_0(Z,X) + U,\\

\\Z = m_0(X) + V,\\

with \\E\[\nu\|Z,X\]=0\\, \\E\[U\|Z,X\]=0\\ and \\E\[V\|X\]=0\\. The
target parameter of interest in this model is the local average
treatment effect (LATE),

\\\theta_0 = \frac{E\[g_0(1,X)\] - E\[g_0(0,X)\]}{E\[r_0(1,X)\] -
E\[r_0(0,X)\]}.\\

## See also

Other DoubleML: [`DoubleML`](DoubleML.md),
[`DoubleMLIRM`](DoubleMLIRM.md), [`DoubleMLPLIV`](DoubleMLPLIV.md),
[`DoubleMLPLR`](DoubleMLPLR.md), [`DoubleMLSSM`](DoubleMLSSM.md)

## Super class

[`DoubleML::DoubleML`](DoubleML.md) -\> `DoubleMLIIVM`

## Active bindings

- `subgroups`:

  (named `list(2)`)  
  Named `list(2)` with options to adapt to cases with and without the
  subgroups of always-takers and never-takes. The entry
  `always_takers`(`logical(1)`) speficies whether there are always
  takers in the sample. The entry `never_takers` (`logical(1)`)
  speficies whether there are never takers in the sample.

- `trimming_rule`:

  (`character(1)`)  
  A `character(1)` specifying the trimming approach.

- `trimming_threshold`:

  (`numeric(1)`)  
  The threshold used for timming.

## Methods

### Public methods

- [`DoubleMLIIVM$new()`](#method-DoubleMLIIVM-new)

- [`DoubleMLIIVM$clone()`](#method-DoubleMLIIVM-clone)

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

    DoubleMLIIVM$new(
      data,
      ml_g,
      ml_m,
      ml_r,
      n_folds = 5,
      n_rep = 1,
      score = "LATE",
      subgroups = list(always_takers = TRUE, never_takers = TRUE),
      dml_procedure = "dml2",
      trimming_rule = "truncate",
      trimming_threshold = 1e-12,
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
  `ml_g` refers to the nuisance function \\g_0(Z,X) = E\[Y\|X,Z\]\\.

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
  `ml_m` refers to the nuisance function \\m_0(X) = E\[Z\|X\]\\.

- `ml_r`:

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
  `ml_r` refers to the nuisance function \\r_0(Z,X) = E\[D\|X,Z\]\\.

- `n_folds`:

  (`integer(1)`)  
  Number of folds. Default is `5`.

- `n_rep`:

  (`integer(1)`)  
  Number of repetitions for the sample splitting. Default is `1`.

- `score`:

  (`character(1)`, `function()`)  
  A `character(1)` (`"LATE"` is the only choice) specifying the score
  function. If a `function()` is provided, it must be of the form
  `function(y, z, d, g0_hat, g1_hat, m_hat, r0_hat, r1_hat, smpls)` and
  the returned output must be a named
  [`list()`](https://rdrr.io/r/base/list.html) with elements `psi_a` and
  `psi_b`. Default is `"LATE"`.

- `subgroups`:

  (named `list(2)`)  
  Named `list(2)` with options to adapt to cases with and without the
  subgroups of always-takers and never-takes. The entry
  `always_takers`(`logical(1)`) speficies whether there are always
  takers in the sample. The entry `never_takers` (`logical(1)`)
  speficies whether there are never takers in the sample. Default is
  `list(always_takers = TRUE, never_takers = TRUE)`.

- `dml_procedure`:

  (`character(1)`)  
  A `character(1)` (`"dml1"` or `"dml2"`) specifying the double machine
  learning algorithm. Default is `"dml2"`.

- `trimming_rule`:

  (`character(1)`)  
  A `character(1)` (`"truncate"` is the only choice) specifying the
  trimming approach. Default is `"truncate"`.

- `trimming_threshold`:

  (`numeric(1)`)  
  The threshold used for timming. Default is `1e-12`.

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

    DoubleMLIIVM$clone(deep = FALSE)

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
#> 
#> Attaching package: ‘data.table’
#> The following object is masked from ‘package:base’:
#> 
#>     %notin%
set.seed(2)
ml_g = lrn("regr.ranger",
  num.trees = 100, mtry = 20,
  min.node.size = 2, max.depth = 5)
ml_m = lrn("classif.ranger",
  num.trees = 100, mtry = 20,
  min.node.size = 2, max.depth = 5)
ml_r = ml_m$clone()
obj_dml_data = make_iivm_data(
  theta = 0.5, n_obs = 1000,
  alpha_x = 1, dim_x = 20)
dml_iivm_obj = DoubleMLIIVM$new(obj_dml_data, ml_g, ml_m, ml_r)
dml_iivm_obj$fit()
dml_iivm_obj$summary()
#> Estimates and significance testing of the effect of target variables
#>   Estimate. Std. Error t value Pr(>|t|)  
#> d    0.5398     0.2149   2.511    0.012 *
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
ml_r = ml_m$clone()
obj_dml_data = make_iivm_data(
  theta = 0.5, n_obs = 1000,
  alpha_x = 1, dim_x = 20)
dml_iivm_obj = DoubleMLIIVM$new(obj_dml_data, ml_g, ml_m, ml_r)
param_grid = list(
  "ml_g" = paradox::ps(
    cp = paradox::p_dbl(lower = 0.01, upper = 0.02),
    minsplit = paradox::p_int(lower = 1, upper = 2)),
  "ml_m" = paradox::ps(
    cp = paradox::p_dbl(lower = 0.01, upper = 0.02),
    minsplit = paradox::p_int(lower = 1, upper = 2)),
  "ml_r" = paradox::ps(
    cp = paradox::p_dbl(lower = 0.01, upper = 0.02),
    minsplit = paradox::p_int(lower = 1, upper = 2)))
# minimum requirements for tune_settings
tune_settings = list(
  terminator = mlr3tuning::trm("evals", n_evals = 5),
  algorithm = mlr3tuning::tnr("grid_search", resolution = 5))
dml_iivm_obj$tune(param_set = param_grid, tune_settings = tune_settings)
dml_iivm_obj$fit()
dml_iivm_obj$summary()
} # }
```
