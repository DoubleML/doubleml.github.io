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

bart_sens<-function(usek, name_output, cov=c()){
  Q_cols_drop=c(c("iqsb.36"),cov)
  folds <- createFolds(y=usek$iqsb.36, k=10, returnTrain = TRUE)
  all_preds <- rep(NA, dim(usek)[[1]])
  
  for (n in 1:length(folds)){
    cat(n)
    usek_train <- usek[folds[[n]],]
    usek_test <- usek[-folds[[n]],]
    xt_Q <- as.matrix(usek_train[,!names(usek_train) %in% Q_cols_drop])
    y_Q <- as.numeric(usek_train[,"iqsb.36"])
    x_test_Q <- as.matrix(usek_test[,!names(usek_test) %in% Q_cols_drop])
    bart_Q.tot <- bart(x.train=xt_Q,   y.train=y_Q, x.test=x_test_Q)
    preds<-bart_Q.tot['yhat.test'][[1]]
    preds<-colMeans(preds)
    all_preds[-folds[[n]]]<-preds
  }
  stopifnot(sum(is.na(all_preds))==0)
  y<-as.numeric(usek[, "iqsb.36"])
  t<-as.numeric(usek[, "dose400"])
  out<-cbind(all_preds, t, y)
  colnames(out)<-c('Q', 't', 'y')
  if(length(cov)!=0){
    colnames(out)<-c('Qhat', 't','y')
  }
  write.csv(out, file= paste0('../out/ihdp_bart_kfold_grouped/', name_output, '.csv'), row.names = FALSE)
}

bart_sens(usek, 'input_df')

loc<-c("site1", "site2", "site3", "site4", "site5", "site6", "site7", "site8")
socioeconomic<-c('mom.lths', "mom.hs", "mom.scoll", "mom.coll", "b.marry", "livwho", "language", "momblack", "momhisp", "momwhite")
mom_preg<-c("birth.o", "parity", "moreprem", "cigs", "alcohol", "mlt.birt", "whenpren", "drugs", "workdur.imp")
baby<-c("bw", "nnhealth", "female")
mom_age<-c("momage")
others<-c("ppvt.imp", "bwg", "othstudy")

stopifnot(setequal(covs, c(loc, socioeconomic, mom_preg, baby, mom_age, others)))
combined_covs=list(loc, socioeconomic, mom_preg, baby, mom_age, others)
combined_covs_names=c("location", "socioeconomic", "mom_preg", "baby", "mom_age", "others")


for (i in 1:length(combined_covs)){
  bart_sens(usek, paste0('covariates/', combined_covs_names[[i]]), cov=combined_covs[[i]])
}

#ADD TREATMENT
bart_sens(usek, paste0('covariates/treatment'), cov=c('dose400'))





