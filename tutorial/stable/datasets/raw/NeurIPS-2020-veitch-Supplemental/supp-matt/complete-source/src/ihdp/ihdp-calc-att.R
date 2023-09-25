library(BayesTree)
library(caret)
setwd("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
load("../data/ihdp-raw/ihdp-hill-2011/data/example.data")
set.seed(3847293)
covs.cont=c("bw","momage","nnhealth","birth.o","parity","moreprem","cigs","alcohol","ppvt.imp")
covs.cat=c("bwg","female","mlt.birt","b.marry","livwho","language","whenpren","drugs","othstudy","mom.lths","mom.hs","mom.coll","mom.scoll","site1","site2","site3","site4","site5","site6","site7","site8","momblack","momhisp","momwhite","workdur.imp")
covs=c(covs.cont,covs.cat)
ncovs=length(covs)

usek = na.omit(ihdp[!(ihdp$treat==1 & ihdp$dose400==0),c("iqsb.36","dose400",covs)])


calc_att<-function(data, treat_col, out_col){
  Q_cols_drop=c(out_col)
  folds <- createFolds(y=data[, out_col], k=10, returnTrain = TRUE)
  treat_out<-c()
  preds_out<-c()
  real_treat_out<-c()
  
  for (n in 1:length(folds)){
    data_train <- data[folds[[n]],]
    data_test <- data[-folds[[n]],]
    xt_Q <- as.matrix(data_train[,!names(data_train) %in% Q_cols_drop])
    y_Q <- as.numeric(data_train[,out_col])
    x_test_Q <- as.matrix(data_test[,!names(data_test) %in% Q_cols_drop])
    xtest_Q0=cbind(x_test_Q)
    xtest_Q1=cbind(x_test_Q)
    real_treat_out <- c(real_treat_out, xtest_Q0[, treat_col])
    real_treat_out <- c(real_treat_out, xtest_Q1[, treat_col])
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
  att_df=cbind(real_treat_out, treat_out, preds_out)
  att_df<-as.data.frame(att_df)
  att_df<-att_df[which(att_df$real_treat_out==1),]
  att<-mean(att_df[which(att_df$treat_out==1), 'preds_out']-att_df[which(att_df$treat_out==0), 'preds_out'])
  sd_y<-sd(data[, out_col])
  cat(att, sd_y)
}


calc_att(usek, "dose400", "iqsb.36")

socioeconomic<-c('mom.lths', "mom.hs", "mom.scoll", "mom.coll", "b.marry", "livwho", "language", "momblack", "momhisp", "momwhite")
usek_min_socio<-usek[, !names(usek) %in% socioeconomic]

calc_att(usek_min_socio, "dose400", "iqsb.36")

