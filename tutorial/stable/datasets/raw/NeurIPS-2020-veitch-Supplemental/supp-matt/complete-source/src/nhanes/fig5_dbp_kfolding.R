library(BayesTree)
library(foreign)
library(caret)
setwd("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
data=read.dta("../data/nhanes-raw/sensitivity/NHANES/NHANES3hbp_dbp.dta")

covs<-c("white", "black", "hisp", "female", "age_mo", "hhsize", "edu", "married", "widowed", "divorced", "separated", "income", "packyr", "bmi", "pulse", "sodium", "potassium", "r_sodipota", "alcohol", "insurance", "together")
to_drop<-c("ave_sbp", "d_ctrl", "num_aht", "trt_sbp")
data<-data[, !names(data) %in% to_drop]

bart_sens<-function(data,  out_dir, out_name, treat_col, out_col, cov=c()){
  Q_cols_drop=c(c(out_col),cov)
  folds <- createFolds(y=data[, out_col], k=10, returnTrain = TRUE)
  all_preds <- rep(NA, dim(data)[[1]])

  
  for (n in 1:length(folds)){
    data_train <- data[folds[[n]],]
    data_test <- data[-folds[[n]],]
    xt_Q <- as.matrix(data_train[,!names(data_train) %in% Q_cols_drop])
    y_Q <- as.numeric(data_train[,out_col])
    x_test_Q <- as.matrix(data_test[,!names(data_test) %in% Q_cols_drop])
    bart_Q.tot <- bart(x.train=xt_Q,   y.train=y_Q, x.test=x_test_Q)
    preds<-bart_Q.tot['yhat.test'][[1]]
    preds<-colMeans(preds)
    all_preds[-folds[[n]]]<-preds
  }
  cat(sum(is.na(all_preds)))
  y<-as.numeric(data[, out_col])
  t<-as.numeric(data[, treat_col])
  out<-cbind(all_preds, t, y)
  colnames(out)<-c('Q', 't', 'y')
  if(length(cov)!=0){
    colnames(out)<-c('Qhat', 't','y')
  }
  write.csv(out, file= paste0(out_dir, out_name, '.csv'), row.names = FALSE)
}
  

calc_ate<-function(data, treat_col, out_col){
  Q_cols_drop=c(out_col)
  folds <- createFolds(y=data[, out_col], k=10, returnTrain = TRUE)
  treat_out<-c()
  preds_out<-c()
  
  for (n in 1:length(folds)){
    data_train <- data[folds[[n]],]
    data_test <- data[-folds[[n]],]
    xt_Q <- as.matrix(data_train[,!names(data_train) %in% Q_cols_drop])
    y_Q <- as.numeric(data_train[,out_col])
    x_test_Q <- as.matrix(data_test[,!names(data_test) %in% Q_cols_drop])
    xtest_Q0=cbind(x_test_Q)
    xtest_Q1=cbind(x_test_Q)
    xtest_Q0[, treat_col]=0
    xtest_Q1[, treat_col]=1
    x_test=rbind(xtest_Q0, xtest_Q1)
    treat_out <- c(treat_out, x_test[, treat_col])
    cat(dim(xt_Q), 'yy', dim(y_Q), 'zz', dim(x_test))
    bart_Q.tot <- bart(x.train=xt_Q,   y.train=y_Q, x.test = x_test)
    preds<-bart_Q.tot['yhat.test'][[1]]
    preds<-colMeans(preds)
    preds_out<-c(preds_out, preds)
  }
  ate_df=cbind(treat_out, preds_out)
  ate_df<-as.data.frame(ate_df)
  ate<-mean(ate_df[which(ate_df$treat_out==1), 'preds_out']-ate_df[which(ate_df$treat_out==0), 'preds_out'])
  sd_y<-sd(data[, out_col])
  cat(ate, sd_y)
}

bart_sens(data, '../out/nhanes_dbp_bart_kfold_grouped/', 'input_df', 'trt_dbp', 'ave_dbp')

habits<-c('alcohol', 'packyr')
race<-c('black', 'hisp', 'white')
wealth<-c('income', 'insurance')
bloodwork<-c('bmi', 'potassium', 'pulse', 'r_sodipota', 'sodium')
social<-c('divorced', 'hhsize', 'married', 'widowed', 'together', 'separated')
gender<-c('female')
age<-c('age_mo')
education<-c('edu')
socioeconomic<-c(social, wealth, habits, race, education)


combined_covs=list(habits, race, wealth, bloodwork, social, gender, age, education, socioeconomic)
combined_covs_names=c("habits", "race", "wealth", "bloodwork", "social", "gender", "age", "education", "socioeconomic")
for (i in 1:length(combined_covs)){
  bart_sens(data, '../out/nhanes_dbp_bart_kfold_grouped/', paste0('covariates/', combined_covs_names[[i]]), 'trt_dbp', 'ave_dbp', cov=combined_covs[[i]])
}


#DO TREATMENT
bart_sens(data, '../out/nhanes_dbp_bart_kfold_grouped/', paste0('covariates/', 'treatment'), 'trt_dbp', 'ave_dbp', cov='trt_dbp')

#MINUS AGE
data_noage<-data[,!names(data) %in% c("age_mo")]
bart_sens(data_noage, '../out/nhanes_dbp_bart_kfold_grouped_noage/', 'input_df', 'trt_dbp', 'ave_dbp')
combined_covs_noage=list(bloodwork, gender, socioeconomic)
combined_covs_names_noage=c("bloodwork", "gender", "socioeconomic")
for (i in 1:length(combined_covs_noage)){
  bart_sens(data, '../out/nhanes_dbp_bart_kfold_grouped_noage/', paste0('covariates/', combined_covs_names_noage[[i]]), 'trt_dbp', 'ave_dbp', cov=combined_covs_noage[[i]])
}


for (i in 1:length(combined_covs)){
  bart_sens(data_noage, '../out/nhanes_dbp_bart_kfold_grouped_noage/', paste0('covariates/', combined_covs_names[[i]]), 'trt_dbp', 'ave_dbp', cov=combined_covs[[i]])
}
bart_sens(data, '../out/nhanes_dbp_bart_kfold_grouped_noage/', paste0('covariates/', 'treatment'), 'trt_dbp', 'ave_dbp', cov='trt_dbp')


#ATE
calc_ate(data, 'trt_dbp', 'ave_dbp')
#-2.332649 13.73849
calc_ate(data_noage, 'trt_dbp', 'ave_dbp')
#-2.856636 13.73849


