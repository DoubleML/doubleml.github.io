# Getting Started with DoubleML

The purpose of the following case-studies is to demonstrate the core
functionalities of `DoubleML`.

## Installation

The **DoubleML** package for R can be downloaded using (requires
previous installation of the [`remotes`
package](https://remotes.r-lib.org/index.html)).

``` r

remotes::install_github("DoubleML/doubleml-for-r")
```

Load the package after completed installation.

``` r

library(DoubleML)
```

The python package `DoubleML` is available via the github repository.
For more information, please visit our user guide.

## Data

For our case study we download the Bonus data set from the Pennsylvania
Reemployment Bonus experiment and as a second example we simulate data
from a partially linear regression model.

``` r

library(DoubleML)

# Load bonus data
df_bonus = fetch_bonus(return_type = "data.table")
head(df_bonus)
```

    ##    inuidur1 female black othrace  dep1  dep2    q2    q3    q4    q5    q6
    ##       <num>  <num> <num>   <num> <num> <num> <num> <num> <num> <num> <num>
    ## 1: 2.890372      0     0       0     0     1     0     0     0     1     0
    ## 2: 0.000000      0     0       0     0     0     0     0     0     1     0
    ## 3: 3.295837      0     0       0     0     0     0     0     1     0     0
    ## 4: 2.197225      0     0       0     0     0     0     1     0     0     0
    ## 5: 3.295837      0     0       0     1     0     0     0     0     1     0
    ## 6: 3.295837      1     0       0     0     0     0     0     0     1     0
    ##    agelt35 agegt54 durable  lusd  husd    tg
    ##      <num>   <num>   <num> <num> <num> <num>
    ## 1:       0       0       0     0     1     0
    ## 2:       0       0       0     1     0     0
    ## 3:       0       0       0     1     0     0
    ## 4:       1       0       0     0     0     1
    ## 5:       0       1       1     1     0     0
    ## 6:       0       1       0     1     0     0

``` r

# Simulate data
set.seed(3141)
n_obs = 500
n_vars = 100
theta = 3
X = matrix(rnorm(n_obs * n_vars), nrow = n_obs, ncol = n_vars)
d = X[, 1:3] %*% c(5, 5, 5) + rnorm(n_obs)
y = theta * d + X[, 1:3] %*% c(5, 5, 5) + rnorm(n_obs)
```

## The causal model

``` math
\begin{align*}
Y = D \theta_0 + g_0(X) + \zeta, & &\mathbb{E}(\zeta | D,X) = 0, \\
D = m_0(X) + V, & &\mathbb{E}(V | X) = 0,
\end{align*}
```
where $`Y`$ is the outcome variable and $`D`$ is the policy variable of
interest. The high-dimensional vector $`X = (X_1, \ldots, X_p)`$
consists of other confounding covariates, and $`\zeta`$ and $`V`$ are
stochastic errors.

## The data-backend `DoubleMLData`

`DoubleML` provides interfaces to objects of class
[`data.table`](https://rdatatable.gitlab.io/data.table/) as well as R
base classes `data.frame` and `matrix`. Details on the data-backend and
the interfaces can be found in the user guide. The `DoubleMLData` class
serves as data-backend and can be initialized from a dataframe by
specifying the column `y_col="inuidur1"` serving as outcome variable
$`Y`$, the column(s) `d_cols = "tg"` serving as treatment variable $`D`$
and the columns
`x_cols=c("female", "black", "othrace", "dep1", "dep2", "q2", "q3", "q4", "q5", "q6", "agelt35", "agegt54", "durable", "lusd", "husd")`
specifying the confounders. Alternatively a matrix interface can be used
as shown below for the simulated data.

``` r

# Specify the data and variables for the causal model
dml_data_bonus = DoubleMLData$new(df_bonus,
  y_col = "inuidur1",
  d_cols = "tg",
  x_cols = c("female", "black", "othrace", "dep1", "dep2",
    "q2", "q3", "q4", "q5", "q6", "agelt35", "agegt54",
    "durable", "lusd", "husd"))
print(dml_data_bonus)
```

    ## ================= DoubleMLData Object ==================
    ## 
    ## 
    ## ------------------ Data summary      ------------------
    ## Outcome variable: inuidur1
    ## Treatment variable(s): tg
    ## Covariates: female, black, othrace, dep1, dep2, q2, q3, q4, q5, q6, agelt35, agegt54, durable, lusd, husd
    ## Instrument(s): 
    ## Selection variable: 
    ## No. Observations: 5099

``` r

# matrix interface to DoubleMLData
dml_data_sim = double_ml_data_from_matrix(X = X, y = y, d = d)
dml_data_sim
```

    ## ================= DoubleMLData Object ==================
    ## 
    ## 
    ## ------------------ Data summary      ------------------
    ## Outcome variable: y
    ## Treatment variable(s): d
    ## Covariates: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, X31, X32, X33, X34, X35, X36, X37, X38, X39, X40, X41, X42, X43, X44, X45, X46, X47, X48, X49, X50, X51, X52, X53, X54, X55, X56, X57, X58, X59, X60, X61, X62, X63, X64, X65, X66, X67, X68, X69, X70, X71, X72, X73, X74, X75, X76, X77, X78, X79, X80, X81, X82, X83, X84, X85, X86, X87, X88, X89, X90, X91, X92, X93, X94, X95, X96, X97, X98, X99, X100
    ## Instrument(s): 
    ## Selection variable: 
    ## No. Observations: 500

## Learners to estimate the nuisance models

To estimate our partially linear regression (PLR) model with the double
machine learning algorithm, we first have to specify machine learners to
estimate $`m_0`$ and $`g_0`$. For the bonus data we use a random forest
regression model and for our simulated data from a sparse partially
linear model we use a Lasso regression model. The implementation of
`DoubleML` is based on the meta-packages
[mlr3](https://mlr3.mlr-org.com/) for R. For details on the
specification of learners and their hyperparameters we refer to the user
guide Learners, hyperparameters and hyperparameter tuning.

``` r

library(mlr3)
library(mlr3learners)
# surpress messages from mlr3 package during fitting
lgr::get_logger("mlr3")$set_threshold("warn")

learner = lrn("regr.ranger", num.trees = 500, max.depth = 5, min.node.size = 2)
ml_l_bonus = learner$clone()
ml_m_bonus = learner$clone()

learner = lrn("regr.glmnet", lambda = sqrt(log(n_vars) / (n_obs)))
ml_l_sim = learner$clone()
ml_m_sim = learner$clone()
```

## Cross-fitting, DML algorithms and Neyman-orthogonal score functions

When initializing the object for PLR models `DoubleMLPLR`, we can
further set parameters specifying the resampling:

- The number of folds used for cross-fitting `n_folds` (defaults to
  `n_folds = 5`) as well as
- the number of repetitions when applying repeated cross-fitting `n_rep`
  (defaults to `n_rep = 1`).

Additionally, one can choose between the algorithms `"dml1"` and
`"dml2"` via `dml_procedure` (defaults to `"dml2"`). Depending on the
causal model, one can further choose between different Neyman-orthogonal
score / moment functions. For the PLR model the default score is
`"partialling out"`, i.e.,
``` math
\begin{align}\begin{aligned}\psi(W; \theta, \eta) &:= [Y - \ell(X) - \theta (D - m(X))] [D - m(X)].\end{aligned}\end{align}
```

Note that with this score, we do not estimate $`g_0(X)`$ directly, but
the conditional expectation of $`Y`$ given $`X`$,
$`\ell_0(X) = E[Y|X]`$. The user guide provides details about the
Sample-splitting, cross-fitting and repeated cross-fitting, the Double
machine learning algorithms and the Score functions

## Estimate double/debiased machine learning models

We now initialize `DoubleMLPLR` objects for our examples using default
parameters. The models are estimated by calling the `fit()` method and
we can for example inspect the estimated treatment effect using the
[`summary()`](https://rdrr.io/r/base/summary.html) method. A more
detailed result summary can be obtained via the
[`print()`](https://rdrr.io/r/base/print.html) method. Besides the
`fit()` method `DoubleML` model classes also provide functionalities to
perform statistical inference like `bootstrap()`,
[`confint()`](https://rdrr.io/r/stats/confint.html) and `p_adjust()`,
for details see the user guide Variance estimation, confidence intervals
and boostrap standard errors.

``` r

set.seed(3141)
obj_dml_plr_bonus = DoubleMLPLR$new(dml_data_bonus, ml_l = ml_l_bonus, ml_m = ml_m_bonus)
obj_dml_plr_bonus$fit()
print(obj_dml_plr_bonus)
```

    ## ================= DoubleMLPLR Object ==================
    ## 
    ## 
    ## ------------------ Data summary      ------------------
    ## Outcome variable: inuidur1
    ## Treatment variable(s): tg
    ## Covariates: female, black, othrace, dep1, dep2, q2, q3, q4, q5, q6, agelt35, agegt54, durable, lusd, husd
    ## Instrument(s): 
    ## Selection variable: 
    ## No. Observations: 5099
    ## 
    ## ------------------ Score & algorithm ------------------
    ## Score function: partialling out
    ## DML algorithm: dml2
    ## 
    ## ------------------ Machine learner   ------------------
    ## ml_l: regr.ranger
    ## ml_m: regr.ranger
    ## 
    ## ------------------ Resampling        ------------------
    ## No. folds: 5
    ## No. repeated sample splits: 1
    ## Apply cross-fitting: TRUE
    ## 
    ## ------------------ Fit summary       ------------------
    ##  Estimates and significance testing of the effect of target variables
    ##    Estimate. Std. Error t value Pr(>|t|)  
    ## tg  -0.07561    0.03536  -2.139   0.0325 *
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r

obj_dml_plr_sim = DoubleMLPLR$new(dml_data_sim, ml_l = ml_l_sim, ml_m = ml_m_sim)
obj_dml_plr_sim$fit()
print(obj_dml_plr_sim)
```

    ## ================= DoubleMLPLR Object ==================
    ## 
    ## 
    ## ------------------ Data summary      ------------------
    ## Outcome variable: y
    ## Treatment variable(s): d
    ## Covariates: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, X31, X32, X33, X34, X35, X36, X37, X38, X39, X40, X41, X42, X43, X44, X45, X46, X47, X48, X49, X50, X51, X52, X53, X54, X55, X56, X57, X58, X59, X60, X61, X62, X63, X64, X65, X66, X67, X68, X69, X70, X71, X72, X73, X74, X75, X76, X77, X78, X79, X80, X81, X82, X83, X84, X85, X86, X87, X88, X89, X90, X91, X92, X93, X94, X95, X96, X97, X98, X99, X100
    ## Instrument(s): 
    ## Selection variable: 
    ## No. Observations: 500
    ## 
    ## ------------------ Score & algorithm ------------------
    ## Score function: partialling out
    ## DML algorithm: dml2
    ## 
    ## ------------------ Machine learner   ------------------
    ## ml_l: regr.glmnet
    ## ml_m: regr.glmnet
    ## 
    ## ------------------ Resampling        ------------------
    ## No. folds: 5
    ## No. repeated sample splits: 1
    ## Apply cross-fitting: TRUE
    ## 
    ## ------------------ Fit summary       ------------------
    ##  Estimates and significance testing of the effect of target variables
    ##   Estimate. Std. Error t value Pr(>|t|)    
    ## d   2.98094    0.05871   50.78   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
