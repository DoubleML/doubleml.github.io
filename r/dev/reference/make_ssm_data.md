# Generates data from a sample selection model (SSM).

The data generating process is defined as:

## Usage

``` r
make_ssm_data(
  n_obs = 8000,
  dim_x = 100,
  theta = 1,
  mar = TRUE,
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

- mar:

  (`logical(1)`)  
  Indicates whether missingness at random holds.

- return_type:

  (`character(1)`)  
  If `"DoubleMLData"`, returns a `DoubleMLData` object. If
  `"data.frame"` returns a
  [`data.frame()`](https://rdrr.io/r/base/data.frame.html). If
  `"data.table"` returns a
  [`data.table()`](https://rdrr.io/pkg/data.table/man/data.table.html).
  Default is `"DoubleMLData"`.

## Value

Depending on the `return_type`, returns an object or set of objects as
specified.

## Details

\$\$ y_i = \theta d_i + x_i' \beta + u_i,\$\$

\$\$s_i = 1\lbrace d_i + \gamma z_i + x_i' \beta + v_i \> 0 \rbrace,\$\$

\$\$d_i = 1\lbrace x_i' \beta + w_i \> 0 \rbrace,\$\$

with \\y_i\\ being observed if \\s_i = 1\\ and covariates \\x_i \sim
\mathcal{N}(0, \Sigma^2_x)\\, where \\\Sigma^2_x\\ is a matrix with
entries \\\Sigma\_{kj} = 0.5^{\|j-k\|}\\. \\\beta\\ is a `dim_x`-vector
with entries \\\beta_j=\frac{0.4}{j^2}\\ \\z_i \sim \mathcal{N}(0, 1)\\,
\\(u_i,v_i) \sim \mathcal{N}(0, \Sigma^2\_{u,v})\\, \\w_i \sim
\mathcal{N}(0, 1)\\.

The data generating process is inspired by a process used in the
simulation study (see Appendix E) of Bia, Huber and Lafférs (2023).

## References

Michela Bia, Martin Huber & Lukáš Lafférs (2023) Double Machine Learning
for Sample Selection Models, Journal of Business & Economic Statistics,
DOI: 10.1080/07350015.2023.2271071
