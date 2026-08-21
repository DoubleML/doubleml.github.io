# Generates data from a interactive regression (IRM) model.

Generates data from a interactive regression (IRM) model. The data
generating process is defined as

\\d_i = 1\left\lbrace \frac{\exp(c_d x_i' \beta)}{1+\exp(c_d x_i'
\beta)} \> v_i \right\rbrace,\\

\\ y_i = \theta d_i + c_y x_i' \beta d_i + \zeta_i,\\

with \\v_i \sim \mathcal{U}(0,1)\\, \\\zeta_i \sim \mathcal{N}(0,1)\\
and covariates \\x_i \sim \mathcal{N}(0, \Sigma)\\, where \\\Sigma\\ is
a matrix with entries \\\Sigma\_{kj} = 0.5^{\|j-k\|}\\. \\\beta\\ is a
`dim_x`-vector with entries \\\beta_j = \frac{1}{j^2}\\ and the
constancts \\c_y\\ and \\c_d\\ are given by

\\ c_y = \sqrt{\frac{R_y^2}{(1-R_y^2) \beta' \Sigma \beta}},\\

\\c_d = \sqrt{\frac{(\pi^2 /3) R_d^2}{(1-R_d^2) \beta' \Sigma \beta}}.\\

The data generating process is inspired by a process used in the
simulation experiment (see Appendix P) of Belloni et al. (2017).

## Usage

``` r
make_irm_data(
  n_obs = 500,
  dim_x = 20,
  theta = 0,
  R2_d = 0.5,
  R2_y = 0.5,
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

- theta:

  (`numeric(1)`)  
  The value of the causal parameter.

- R2_d:

  (`numeric(1)`)  
  The value of the parameter \\R_d^2\\.

- R2_y:

  (`numeric(1)`)  
  The value of the parameter \\R_y^2\\.

- return_type:

  (`character(1)`)  
  If `"DoubleMLData"`, returns a `DoubleMLData` object. If
  `"data.frame"` returns a
  [`data.frame()`](https://rdrr.io/r/base/data.frame.html). If
  `"data.table"` returns a
  [`data.table()`](https://rdrr.io/pkg/data.table/man/data.table.html).
  If `"matrix"` a named [`list()`](https://rdrr.io/r/base/list.html)
  with entries `X`, `y`, `d` and `z` is returned. Every entry in the
  list is a [`matrix()`](https://rdrr.io/r/base/matrix.html) object.
  Default is `"DoubleMLData"`.

## References

Belloni, A., Chernozhukov, V., Fernández-Val, I. and Hansen, C. (2017).
Program Evaluation and Causal Inference With High-Dimensional Data.
Econometrica, 85: 233-298.
