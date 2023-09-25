library(BayesTree)
setwd("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
load("../data/ihdp-raw/ihdp-hill-2011/data/example.data")
set.seed(3847293)
covs.cont=c("bw","momage","nnhealth","birth.o","parity","moreprem","cigs","alcohol","ppvt.imp")
covs.cat=c("bwg","female","mlt.birt","b.marry","livwho","language","whenpren","drugs","othstudy","mom.lths","mom.hs","mom.coll","mom.scoll","site1","site2","site3","site4","site5","site6","site7","site8","momblack","momhisp","momwhite","workdur.imp")
covs=c(covs.cont,covs.cat)
ncovs=length(covs)

usek = na.omit(ihdp[!(ihdp$treat==1 & ihdp$dose400==0),c("iqsb.36","dose400",covs)])
write.csv(usek, '../data/ihdp-cleaned/usek.csv', row.names = FALSE)
# treat 1-dose1
# treat 0- dose0
#Q = yhat.train.mean(). make predictions on treated (keeping them as 1)
#g = g for train. make predictions on treated (keeping them as 1)

# important to excluse the treated with "low" dose since "highdose=0"
# for them means something very different than it does for those in
# then control group
bart_sens<-function(usek, name_output, cov=c()){
  Q_cols_drop=c(c("iqsb.36"),cov)
  xt_Q=as.matrix(usek[,!names(usek) %in% Q_cols_drop])
  # xp_Q=as.matrix(usek[usek$dose400==1,!names(usek) %in% Q_cols_drop])
  y_Q=as.numeric(usek[,"iqsb.36"])
  
  g_cols_drop=c(c("iqsb.36", "dose400"),cov)
  xt_g=as.matrix(usek[,!names(usek) %in% g_cols_drop])
  # xp_g=as.matrix(usek[usek$dose400==1, !names(usek) %in% g_cols_drop])
  y_g= as.numeric(usek[,"dose400"])
  bart_Q.tot <- bart(x.train=xt_Q,   y.train=y_Q)
  bart_g.tot <- bart(x.train=xt_g,   y.train=y_g)
  Q_preds=bart_Q.tot['yhat.train'][[1]]
  # you need to do pnorm to get probability values
  g_preds=pnorm(bart_g.tot['yhat.train'][[1]])
  Q_preds=colMeans(Q_preds)
  g_preds=colMeans(g_preds)
  
  y=as.numeric(usek[, "iqsb.36"])
  t=as.numeric(usek[, "dose400"])
  out=cbind(g_preds, Q_preds, t, y)
  colnames(out)<-c('g', 'Q', 't', 'y')
  if(length(cov)!=0){
    colnames(out)<-c('ghat', 'Qhat', 't','y')
  }
  write.csv(out, file= paste0('../out/ihdp_bart_grouped/', name_output, '.csv'), row.names = FALSE)
}

loc=c("site1", "site2", "site3", "site4", "site5", "site6", "site7", "site8")
mom_ed=c('mom.lths', "mom.hs", "mom.scoll")
mom_habits=c("booze", "drugs", "cig")
mom_race=c("momblack", "momhisp", "momwhite")
baby_health=c("bw", "b.head", "preterm", "birth.o", "nnhealth")

bart_sens(usek, 'input_df')

for (covariate in c(covs, 'treatment')){
  bart_sens(usek, paste0('covariates/', covariate), cov=c(covariate))
}
