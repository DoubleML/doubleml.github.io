library(BayesTree)
setwd("C:/Users/light/Dropbox (Personal)/Projects/sense_and_sensitivity/src")
data=read.csv("../data/ihdp-raw/ihdp_csv_1-1000/csv/ihdp_npci_1.csv")

#used seed from Jill's sim code R file
set.seed(2659232)
# Note that the cig and first columns may have been interchanged since naming in Hill's README is different from what appears in Hill's code. Below is from code
covs.cont=c("bw","b.head","preterm","birth.o","nnhealth","momage")
covs.cat=c("sex","twin","b.marr","mom.lths","mom.hs",	"mom.scoll","cig","first","booze","drugs","work.dur","prenatal","ark","ein","har","mia","pen","tex","was")
covs=c(covs.cont,covs.cat)
ncovs=length(covs)

names(data)<-c(c("treatment", "y_factual", "y_cfactual", "mu0", "mu1"), covs)
to_drop<-c("y_cfactual", "mu0", "mu1")
data<-data[, !names(data) %in% to_drop]
write.csv(data, "../data/ihdp-cleaned/ihdp_npci_1_cleaned.csv", row.names = FALSE)

bart_sens<-function(data, name_output, cov=c()){
  Q_cols_drop=c(c("y_factual"),cov)
  xt_Q=as.matrix(data[,!names(data) %in% Q_cols_drop])
  # xp_Q=as.matrix(data[data$treatment==1,!names(data) %in% Q_cols_drop])
  y_Q=as.numeric(data[,"y_factual"])
  
  g_cols_drop=c(c("y_factual", "treatment"),cov)
  xt_g=as.matrix(data[,!names(data) %in% g_cols_drop])
  # xp_g=as.matrix(data[data$treatment==1, !names(data) %in% g_cols_drop])
  y_g= as.numeric(data[,"treatment"])
  bart_Q.tot <- bart(x.train=xt_Q,   y.train=y_Q)
  bart_g.tot <- bart(x.train=xt_g,   y.train=y_g)
  Q_preds=bart_Q.tot['yhat.train'][[1]]
  # you need to do pnorm to get probability values
  g_preds=pnorm(bart_g.tot['yhat.train'][[1]])
  Q_preds=colMeans(Q_preds)
  g_preds=colMeans(g_preds)
  
  y=as.numeric(data[, "y_factual"])
  t=as.numeric(data[, "treatment"])
  out=cbind(g_preds, Q_preds, t, y)
  colnames(out)<-c('g', 'Q', 't', 'y')
  if(length(cov)!=0){
    colnames(out)<-c('ghat', 'Qhat', 't','y')
  }
  write.csv(out, file= paste0('../out/ihdp_bart_sim/', name_output, '.csv'), row.names = FALSE)
}

bart_sens(data, 'input_df')

for (covariate in c(covs, 'treatment')){
  bart_sens(data, paste0('covariates/', covariate), cov=c(covariate))
}

loc=c("ark","ein","har","mia","pen","tex","was")
mom_ed=c('mom.lths', "mom.hs", "mom.scoll")
habits=c("booze", "drugs", "cig")
baby_health=c("bw", "b.head", "preterm", "birth.o", "nnhealth")

combined_covs=list(loc, mom_ed, habits, baby_health)
combined_covs_names=c("loc", "mom_ed", "habits", "baby_health")
for (i in 1:length(combined_covs)){
  bart_sens(data, paste0('covariates/', combined_covs_names[[i]]), cov=combined_covs[[i]])
}
