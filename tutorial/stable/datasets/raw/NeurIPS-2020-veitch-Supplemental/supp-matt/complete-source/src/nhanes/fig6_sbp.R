library(BayesTree)
library(foreign)
setwd("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
data=read.dta("../data/nhanes-raw/sensitivity/NHANES/NHANES3hbp_sbp.dta")

covs<-c("white", "black", "hisp", "female", "age_mo", "hhsize", "edu", "married", "widowed", "divorced", "separated", "income", "packyr", "bmi", "pulse", "sodium", "potassium", "r_sodipota", "alcohol", "insurance", "together")
to_drop<-c("ave_dbp", "d_ctrl", "num_aht", "trt_dbp")
data<-data[, !names(data) %in% to_drop]
write.csv(data, "../data/nhanes-cleaned/hbp_sbp.csv", row.names = FALSE)

bart_sens<-function(data,  out_dir, out_name, treat_col, out_col, cov=c()){
  Q_cols_drop=c(c(out_col),cov)
  xt_Q=as.matrix(data[,!names(data) %in% Q_cols_drop])
  y_Q=as.numeric(data[,out_col])
  g_cols_drop=c(c(out_col, treat_col),cov)
  xt_g=as.matrix(data[,!names(data) %in% g_cols_drop])
  y_g= as.numeric(data[,treat_col])
  bart_Q.tot <- bart(x.train=xt_Q,   y.train=y_Q)
  bart_g.tot <- bart(x.train=xt_g,   y.train=y_g)
  Q_preds=bart_Q.tot['yhat.train'][[1]]
  # you need to do pnorm to get probability values
  g_preds=pnorm(bart_g.tot['yhat.train'][[1]])
  Q_preds=colMeans(Q_preds)
  g_preds=colMeans(g_preds)
  
  y=as.numeric(data[, out_col])
  t=as.numeric(data[, treat_col])
  out=cbind(g_preds, Q_preds, t, y)
  colnames(out)<-c('g', 'Q', 't', 'y')
  if(length(cov)!=0){
    colnames(out)<-c('ghat', 'Qhat', 't','y')
  }
  write.csv(out, file= paste0(out_dir, out_name, '.csv'), row.names = FALSE)
}

bart_sens(data, '../out/nhanes_sbp_bart/', 'input_df', 'trt_sbp', 'ave_sbp')

for (covariate in c(covs, 'trt_sbp')){
  bart_sens(data, '../out/nhanes_sbp_bart/', paste0('covariates/', covariate), 'trt_sbp', 'ave_sbp', cov=c(covariate))
}
