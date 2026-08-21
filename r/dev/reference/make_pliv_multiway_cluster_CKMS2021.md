# Generates data from a partially linear IV regression model with multiway cluster sample used in Chiang et al. (2021).

Generates data from a partially linear IV regression model with multiway
cluster sample used in Chiang et al. (2021). The data generating process
is defined as

\\Z\_{ij} = X\_{ij}' \xi_0 + V\_{ij},\\

\\D\_{ij} = Z\_{ij}' \pi\_{10} + X\_{ij}' \pi\_{20} + v\_{ij},\\

\\Y\_{ij} = D\_{ij} \theta + X\_{ij}' \zeta_0 + \varepsilon\_{ij},\\

with

\\X\_{ij} = (1 - \omega_1^X - \omega_2^X) \alpha\_{ij}^X + \omega_1^X
\alpha\_{i}^X + \omega_2^X \alpha\_{j}^X,\\

\\\varepsilon\_{ij} = (1 - \omega_1^\varepsilon - \omega_2^\varepsilon)
\alpha\_{ij}^\varepsilon + \omega_1^\varepsilon
\alpha\_{i}^\varepsilon + \omega_2^\varepsilon
\alpha\_{j}^\varepsilon,\\

\\v\_{ij} = (1 - \omega_1^v - \omega_2^v) \alpha\_{ij}^v + \omega_1^v
\alpha\_{i}^v + \omega_2^v \alpha\_{j}^v,\\

\\V\_{ij} = (1 - \omega_1^V - \omega_2^V) \alpha\_{ij}^V + \omega_1^V
\alpha\_{i}^V + \omega_2^V \alpha\_{j}^V,\\

and \\\alpha\_{ij}^X, \alpha\_{i}^X, \alpha\_{j}^X \sim \mathcal{N}(0,
\Sigma)\\ where \\\Sigma\\ is a \\p_x \times p_x\\ matrix with entries
\\\Sigma\_{kj} = s_X^{\|j-k\|}\\.

Further

\\\left(\begin{array}{c} \alpha\_{ij}^\varepsilon \\ \alpha\_{ij}^v
\end{array}\right), \left(\begin{array}{c} \alpha\_{i}^\varepsilon \\
\alpha\_{i}^v \end{array}\right), \left(\begin{array}{c}
\alpha\_{j}^\varepsilon \\ \alpha\_{j}^v \end{array}\right) \sim
\mathcal{N}\left(0, \left(\begin{array}{cc} 1 & s\_{\varepsilon v} \\
s\_{\varepsilon v} & 1 \end{array}\right) \right)\\

and \\\alpha\_{ij}^V, \alpha\_{i}^V, \alpha\_{j}^V \sim \mathcal{N}(0,
1)\\.

## Usage

``` r
make_pliv_multiway_cluster_CKMS2021(
  N = 25,
  M = 25,
  dim_X = 100,
  theta = 1,
  return_type = "DoubleMLClusterData",
  ...
)
```

## Arguments

- N:

  (`integer(1)`)  
  The number of observations (first dimension).

- M:

  (`integer(1)`)  
  The number of observations (second dimension).

- dim_X:

  (`integer(1)`)  
  The number of covariates.

- theta:

  (`numeric(1)`)  
  The value of the causal parameter.

- return_type:

  (`character(1)`)  
  If `"DoubleMLClusterData"`, returns a `DoubleMLClusterData` object. If
  `"data.frame"` returns a
  [`data.frame()`](https://rdrr.io/r/base/data.frame.html). If
  `"data.table"` returns a
  [`data.table()`](https://rdrr.io/pkg/data.table/man/data.table.html).
  If `"matrix"` a named [`list()`](https://rdrr.io/r/base/list.html)
  with entries `X`, `y`, `d`, `z` and `cluster_vars` is returned. Every
  entry in the list is a
  [`matrix()`](https://rdrr.io/r/base/matrix.html) object. Default is
  `"DoubleMLClusterData"`.

- ...:

  Additional keyword arguments to set non-default values for the
  parameters \\\pi\_{10}=1.0\\, \\\omega_X = \omega\_{\varepsilon} =
  \omega_V = \omega_v = (0.25, 0.25)\\, \\s_X = s\_{\varepsilon v} =
  0.25\\, or the \\p_x\\-vectors \\\zeta_0 = \pi\_{20} = \xi_0\\ with
  default entries \\\zeta\_{0})\_j = 0.5^j\\.

## Value

A data object according to the choice of `return_type`.

## References

Chiang, H. D., Kato K., Ma, Y. and Sasaki, Y. (2021), Multiway Cluster
Robust Double/Debiased Machine Learning, Journal of Business & Economic
Statistics,
[doi:10.1080/07350015.2021.1895815](https://doi.org/10.1080/07350015.2021.1895815)
, https://arxiv.org/abs/1909.03489.
