library(BayesTree)
library(foreign)
setwd("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
data=read.dta("../data/nhanes-raw/sensitivity/NHANES/NHANES3hbp_dbp.dta")

covs<-c("white", "black", "hisp", "female", "age_mo", "hhsize", "edu", "married", "widowed", "divorced", "separated", "income", "packyr", "bmi", "pulse", "sodium", "potassium", "r_sodipota", "alcohol", "insurance", "together")
to_drop<-c("ave_sbp", "d_ctrl", "num_aht", "trt_sbp")
data<-data[, !names(data) %in% to_drop]
write.csv(data, "../data/nhanes-cleaned/hbp_dbp.csv", row.names = FALSE)

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


calc_ate<-function(data, treat_col, out_col){
  Q_cols_drop=c(out_col)
  xt_Q=as.matrix(data[,!names(data) %in% Q_cols_drop])
  y_Q=as.numeric(data[,out_col])
  
  xt_Q0=cbind(data)
  xt_Q1=cbind(data)
  xt_Q0[, treat_col]=0
  xt_Q1[, treat_col]=1
  x_test=rbind(xt_Q0, xt_Q1)
  x_test_treat_col = x_test[, treat_col]
  x_test=as.matrix(x_test[,!names(x_test) %in% Q_cols_drop])
  
  cat(dim(xt_Q), 'yy', dim(y_Q), 'zz', dim(x_test))
  bart_Q.tot <- bart(x.train=xt_Q,   y.train=y_Q, x.test = x_test)
  Q_preds=bart_Q.tot['yhat.test'][[1]]
  Q_preds=colMeans(Q_preds)
  ate_df=cbind(x_test_treat_col, Q_preds)
  ate_df<-as.data.frame(ate_df)
  ate<-mean(ate_df[which(ate_df$x_test_treat_col==1), 'Q_preds']-ate_df[which(ate_df$x_test_treat_col==0), 'Q_preds'])
  sd_y<-sd(data[, out_col])
  cat(ate, sd_y)
}



data_noage<-data[,!names(data) %in% c("age_mo")]
bart_sens(data_noage, '../out/nhanes_dbp_bart_noage/', 'input_df', 'trt_dbp', 'ave_dbp')
for (covariate in c(covs, 'trt_dbp')){
  bart_sens(data_noage, '../out/nhanes_dbp_bart_noage/', paste0('covariates/', covariate), 'trt_dbp', 'ave_dbp', cov=c(covariate))
}

calc_ate(data, 'trt_dbp', 'ave_dbp')
#-2.221285 13.73849

calc_ate(data_noage, 'trt_dbp', 'ave_dbp')
#-2.838256 13.73849

bart_sens(data, '../out/nhanes_dbp_bart/', 'input_df', 'trt_dbp', 'ave_dbp')

for (covariate in c(covs, 'trt_dbp')){
  bart_sens(data, '../out/nhanes_dbp_bart/', paste0('covariates/', covariate), 'trt_dbp', 'ave_dbp', cov=c(covariate))
}

drugs<-c('alcohol', 'packyr')
race<-c('black', 'hisp', 'white')
wealth<-c('income', 'edu', 'insurance')
bloodwork<-c('bmi', 'potassium', 'pulse', 'r_sodipota', 'sodium')
social<-c('divorced', 'hhsize', 'married', 'widowed', 'together')
socioeconomic<-c(social, wealth, drugs, race)


combined_covs=list(drugs, race, wealth, bloodwork, social, socioeconomic)
combined_covs_names=c("drugs", "race", "wealth", "bloodwork", "social", "socioeconomic")
for (i in 1:length(combined_covs)){
  bart_sens(data, '../out/nhanes_dbp_bart/', paste0('covariates/', combined_covs_names[[i]]), 'trt_dbp', 'ave_dbp', cov=combined_covs[[i]])
}

