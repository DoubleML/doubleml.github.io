# Generates data from a interactive IV regression (IIVM) model.

Generates data from a interactive IV regression (IIVM) model. The data
generating process is defined as

\\d_i = 1\left\lbrace \alpha_x Z + v_i \> 0 \right\rbrace,\\

\\y_i = \theta d_i + x_i' \beta + u_i,\\

\\Z \sim \textstyle{Bernoulli} (0.5)\\ and

\\\left(\begin{array}{c} u_i \\ v_i \end{array} \right) \sim
\mathcal{N}\left(0, \left(\begin{array}{cc} 1 & 0.3 \\ 0.3 & 1
\end{array} \right) \right).\\

The covariates :\\x_i \sim \mathcal{N}(0, \Sigma)\\, where \\\Sigma\\ is
a matrix with entries \\\Sigma\_{kj} = 0.5^{\|j-k\|}\\ and \\\beta\\ is
a `dim_x`-vector with entries \\\beta_j=\frac{1}{j^2}\\.

The data generating process is inspired by a process used in the
simulation experiment of Farbmacher, Gruber and Klaaßen (2020).

## Usage

``` r
make_iivm_data(
  n_obs = 500,
  dim_x = 20,
  theta = 1,
  alpha_x = 0.2,
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

- alpha_x:

  (`numeric(1)`)  
  The value of the parameter \\\alpha_x\\.

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

Farbmacher, H., Guber, R. and Klaaßen, S. (2020). Instrument Validity
Tests with Causal Forests. MEA Discussion Paper No. 13-2020. Available
at SSRN:[doi:10.2139/ssrn.3619201](https://doi.org/10.2139/ssrn.3619201)
.
