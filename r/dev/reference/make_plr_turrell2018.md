# Generates data from a partially linear regression model used in a blog article by Turrell (2018).

Generates data from a partially linear regression model used in a blog
article by Turrell (2018). The data generating process is defined as

\\d_i = m_0(x_i' b) + v_i,\\

\\y_i = \theta d_i + g_0(x_i' b) + u_i,\\

with \\v_i \sim \mathcal{N}(0,1)\\, \\u_i \sim \mathcal{N}(0,1)\\, and
covariates \\x_i \sim \mathcal{N}(0, \Sigma)\\, where \\\Sigma\\ is a
random symmetric, positive-definite matrix generated with
[`clusterGeneration::genPositiveDefMat()`](https://rdrr.io/pkg/clusterGeneration/man/genPositiveDefMat.html).
\\b\\ is a vector with entries \\b_j=\frac{1}{j}\\ and the nuisance
functions are given by

\\m_0(x_i) = \frac{1}{2 \pi} \frac{\sinh(\gamma)}{\cosh(\gamma) -
\cos(x_i-\nu)},\\

\\g_0(x_i) = \sin(x_i)^2.\\

## Usage

``` r
make_plr_turrell2018(
  n_obs = 100,
  dim_x = 20,
  theta = 0.5,
  return_type = "DoubleMLData",
  nu = 0,
  gamma = 1
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

- nu:

  (`numeric(1)`)  
  The value of the parameter \\\nu\\. Default is `0`.

- gamma:

  (`numeric(1)`)  
  The value of the parameter \\\gamma\\. Default is `1`.

## Value

A data object according to the choice of `return_type`.

## References

Turrell, A. (2018), Econometrics in Python part I - Double machine
learning, Markov Wanderer: A blog on economics, science, coding and
data.
<https://aeturrell.com/blog/posts/econometrics-in-python-parti-ml/>.
