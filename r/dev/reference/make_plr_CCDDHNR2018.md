# Generates data from a partially linear regression model used in Chernozhukov et al. (2018)

Generates data from a partially linear regression model used in
Chernozhukov et al. (2018) for Figure 1. The data generating process is
defined as

\\d_i = m_0(x_i) + s_1 v_i,\\

\\y_i = \alpha d_i + g_0(x_i) + s_2 \zeta_i,\\

with \\v_i \sim \mathcal{N}(0,1)\\ and \\\zeta_i \sim
\mathcal{N}(0,1),\\. The covariates are distributed as \\x_i \sim
\mathcal{N}(0, \Sigma)\\, where \\\Sigma\\ is a matrix with entries
\\\Sigma\_{kj} = 0.7^{\|j-k\|}\\. The nuisance functions are given by

\\m_0(x_i) = a_0 x\_{i,1} + a_1
\frac{\exp(x\_{i,3})}{1+\exp(x\_{i,3})},\\

\\g_0(x_i) = b_0 \frac{\exp(x\_{i,1})}{1+\exp(x\_{i,1})} + b_1
x\_{i,3},\\

with \\a_0=1\\, \\a_1=0.25\\, \\s_1=1\\, \\b_0=1\\, \\b_1=0.25\\,
\\s_2=1\\.

## Usage

``` r
make_plr_CCDDHNR2018(
  n_obs = 500,
  dim_x = 20,
  alpha = 0.5,
  return_type = "DoubleMLData"
)
```

## Arguments

- n_obs:

  (`integer(1)`)  
  The number of observations to simulate.

- dim_x:

  (`integer(1)`)  
  The number of covariates.

- alpha:

  (`numeric(1)`)  
  The value of the causal parameter.

- return_type:

  (`character(1)`)  
  If `"DoubleMLData"`, returns a `DoubleMLData` object. If
  `"data.frame"` returns a
  [`data.frame()`](https://rdrr.io/r/base/data.frame.html). If
  `"data.table"` returns a
  [`data.table()`](https://rdrr.io/pkg/data.table/man/data.table.html).
  If `"matrix"` a named [`list()`](https://rdrr.io/r/base/list.html)
  with entries `X`, `y` and `d` is returned. Every entry in the list is
  a [`matrix()`](https://rdrr.io/r/base/matrix.html) object. Default is
  `"DoubleMLData"`.

## Value

A data object according to the choice of `return_type`.

## References

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W. and Robins, J. (2018), Double/debiased machine learning for
treatment and structural parameters. The Econometrics Journal, 21:
C1-C68. [doi:10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097) .
