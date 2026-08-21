# Data set on the Pennsylvania Reemployment Bonus experiment.

Preprocessed data set on the Pennsylvania Reemploymnent Bonus
experiment. The raw data files are preprocessed to reproduce the
examples in Chernozhukov et al. (2020). An internet connection is
required to sucessfully download the data set.

## Usage

``` r
fetch_bonus(return_type = "DoubleMLData", polynomial_features = FALSE)
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

## Value

A data object according to the choice of `return_type`.

## Details

Variable description, based on the supplementary material of
Chernozhukov et al. (2020):

- abdt: chronological time of enrollment of each claimant in the
  Pennsylvania reemployment bonus experiment.

- tg: indicates the treatment group (bonus amount - qualification
  period) of each claimant.

- inuidur1: a measure of length (in weeks) of the first spell of
  unemployment

- inuidur2: a second measure for the length (in weeks) of

- female: dummy variable; it indicates if the claimant's sex is female
  (=1) or male (=0).

- black: dummy variable; it indicates a person of black race (=1).

- hispanic: dummy variable; it indicates a person of hispanic race (=1).

- othrace: dummy variable; it indicates a non-white, non-black,
  not-hispanic person (=1).

- dep1: dummy variable; indicates if the number of dependents of each
  claimant is equal to 1 (=1).

- dep2: dummy variable; indicates if the number of dependents of each
  claimant is equal to 2 (=1).

- q1-q6: six dummy variables indicating the quarter of experiment during
  which each claimant enrolled.

- recall: takes the value of 1 if the claimant answered “yes” when was
  asked if he/she had any expectation to be recalled

- agelt35: takes the value of 1 if the claimant's age is less than 35
  and 0 otherwise.

- agegt54: takes the value of 1 if the claimant's age is more than 54
  and 0 otherwise.

- durable: it takes the value of 1 if the occupation of the claimant was
  in the sector of durable manufacturing and 0 otherwise.

- nondurable: it takes the value of 1 if the occupation of the claimant
  was in the sector of nondurable manufacturing and 0 otherwise.

- lusd: it takes the value of 1 if the claimant filed in Coatesville,
  Reading, or Lancaster and 0 otherwise.

- These three sites were considered to be located in areas characterized
  by low unemployment rate and short duration of unemployment.

- husd: it takes the value of 1 if the claimant filed in Lewistown,
  Pittston, or Scranton and 0 otherwise.

- These three sites were considered to be located in areas characterized
  by high unemployment rate and short duration of unemployment.

- muld: it takes the value of 1 if the claimant filed in
  Philadelphia-North, Philadelphia-Uptown, McKeesport, Erie, or Butler
  and 0 otherwise.

- These three sites were considered to be located in areas characterized
  by moderate unemployment rate and long duration of unemployment."

The supplementary data of the study by Chernozhukov et al. (2018) is
available at
<https://academic.oup.com/ectj/article/21/1/C1/5056401#supplementary-data>.

The supplementary data of the study by Bilias (2000) is available at
<https://www.journaldata.zbw.eu/dataset/sequential-testing-of-duration-data-the-case-of-the-pennsylvania-reemployment-bonus-experiment>.

## References

Bilias Y. (2000), Sequential Testing of Duration Data: The Case of
Pennsylvania ‘Reemployment Bonus’ Experiment. Journal of Applied
Econometrics, 15(6): 575-594.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W. and Robins, J. (2018), Double/debiased machine learning for
treatment and structural parameters. The Econometrics Journal, 21:
C1-C68. [doi:10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097) .

## Examples

``` r
library(DoubleML)
df_bonus = fetch_bonus(return_type = "data.table")
obj_dml_data_bonus = DoubleMLData$new(df_bonus,
  y_col = "inuidur1",
  d_cols = "tg",
  x_cols = c(
    "female", "black", "othrace", "dep1", "dep2",
    "q2", "q3", "q4", "q5", "q6", "agelt35", "agegt54",
    "durable", "lusd", "husd"
  )
)
obj_dml_data_bonus
#> ================= DoubleMLData Object ==================
#> 
#> 
#> ------------------ Data summary      ------------------
#> Outcome variable: inuidur1
#> Treatment variable(s): tg
#> Covariates: female, black, othrace, dep1, dep2, q2, q3, q4, q5, q6, agelt35, agegt54, durable, lusd, husd
#> Instrument(s): 
#> Selection variable: 
#> No. Observations: 5099
```
