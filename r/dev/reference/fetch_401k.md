# Data set on financial wealth and 401(k) plan participation.

Preprocessed data set on financial wealth and 401(k) plan participation.
The raw data files are preprocessed to reproduce the examples in
Chernozhukov et al. (2020). An internet connection is required to
sucessfully download the data set.

## Usage

``` r
fetch_401k(
  return_type = "DoubleMLData",
  polynomial_features = FALSE,
  instrument = FALSE
)
```

## Arguments

- return_type:

  (`character(1)`)  
  If `"DoubleMLData"`, returns a `DoubleMLData` object. If
  `"data.frame"` returns a
  [`data.frame()`](https://rdrr.io/r/base/data.frame.html). If
  `"data.table"` returns a
  [`data.table()`](https://rdrr.io/pkg/data.table/man/data.table.html).
  Default is `"DoubleMLData"`.

- polynomial_features:

  (`logical(1)`)  
  If `TRUE` polynomial freatures are added (see replication file of
  Chernozhukov et al. (2018)).

- instrument:

  (`logical(1)`)  
  If `TRUE`, the returned data object contains the variables `e401` and
  `p401`. If `return_type = "DoubleMLData"`, the variable `e401` is used
  as an instrument for the endogenous treatment variable `p401`. If
  `FALSE`, `p401` is removed from the data set.

## Value

A data object according to the choice of `return_type`.

## Details

Variable description, based on the supplementary material of
Chernozhukov et al. (2020):

- net_tfa: net total financial assets

- e401: = 1 if employer offers 401(k)

- p401: = 1 if individual participates in a 401(k) plan

- age: age

- inc: income

- fsize: family size

- educ: years of education

- db: = 1 if individual has defined benefit pension

- marr: = 1 if married

- twoearn: = 1 if two-earner household

- pira: = 1 if individual participates in IRA plan

- hown: = 1 if home owner

The supplementary data of the study by Chernozhukov et al. (2018) is
available at
<https://academic.oup.com/ectj/article/21/1/C1/5056401#supplementary-data>.

## References

Abadie, A. (2003), Semiparametric instrumental variable estimation of
treatment response models. Journal of Econometrics, 113(2): 231-263.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W. and Robins, J. (2018), Double/debiased machine learning for
treatment and structural parameters. The Econometrics Journal, 21:
C1-C68. [doi:10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097) .
