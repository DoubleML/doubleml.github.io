# Generates data from a partially linear IV regression model used in Chernozhukov, Hansen and Spindler (2015).

Generates data from a partially linear IV regression model used in
Chernozhukov, Hansen and Spindler (2015). The data generating process is
defined as

\\z_i = \Pi x_i + \zeta_i,\\

\\d_i = x_i'\gamma + z_i'\delta + u_i,\\

\\y_i = \alpha d_i + x_i'\beta + \epsilon_i,\\

with

\\\left(\begin{array}{c} \varepsilon_i \\ u_i \\ \zeta_i \\ x_i
\end{array} \right) \sim \mathcal{N}\left(0, \left(\begin{array}{cccc} 1
& 0.6 & 0 & 0 \\ 0.6 & 1 & 0 & 0 \\ 0 & 0 & 0.25 I\_{p_n^z} & 0 \\ 0 & 0
& 0 & \Sigma \end{array} \right) \right)\\

where \\\Sigma\\ is a \\p_n^x \times p_n^x\\ matrix with entries
\\\Sigma\_{kj} = 0.5^{\|j-k\|}\\ and \\I\_{p_n^z}\\ is the \\p^z_n
\times p^z_n\\ identity matrix. \\\beta=\gamma\\ iis a \\p^x_n\\-vector
with entries \\\beta_j = \frac{1}{j^2}\\, \\\delta\\ is a
\\p^z_n\\-vector with entries \\\delta_j = \frac{1}{j^2}\\ and \\\Pi =
(I\_{p_n^z}, O\_{p_n^z \times (p_n^x - p_n^z)})\\.

## Usage

``` r
make_pliv_CHS2015(
  n_obs,
  alpha = 1,
  dim_x = 200,
  dim_z = 150,
  return_type = "DoubleMLData"
)
```

## Arguments

- n_obs:

  (`integer(1)`)  
  The number of observations to simulate.

- alpha:

  (`numeric(1)`)  
  The value of the causal parameter.

- dim_x:

  (`integer(1)`)  
  The number of covariates.

- dim_z:

  (`integer(1)`)  
  The number of instruments.

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

## Value

A data object according to the choice of `return_type`.

## References

Chernozhukov, V., Hansen, C. and Spindler, M. (2015), Post-Selection and
Post-Regularization Inference in Linear Models with Many Controls and
Instruments. American Economic Review: Papers and Proceedings, 105 (5):
486-90.
