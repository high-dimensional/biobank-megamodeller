library(tidyverse)
library(MASS)
library(nlme)
library(cAIC4)
library(sjPlot)
#library(lme4)

df = read.csv('/Users/jruffle/Library/CloudStorage/OneDrive-UniversityCollegeLondon/MSc/MScProject/FIGURE_FILES/metrics_comparison_test.csv')
head(df)

iris = iris

iris_scaled = iris %>%
  group_by(Species) %>%
  mutate(across(contains(c("Width","Length")),scale))

#df = df %>% 
#  group_by(df$Target) %>%
#  mutate(across(contains(c("Balanced.accuracy","r2")),scale))

#lm <- lm(Balanced.accuracy ~ Metadata, data=df)
#lm <- lm(Balanced.accuracy ~ Metadata + Domain, data=df)
#summary(lm)

#lmer_model = lmer(Balanced.accuracy ~ 1 + Metadata + (1|Domain), data=df)
#summary(lmer_model)
#anova(lmer_model)


#m1 <- lme(Balanced.accuracy~Metadata+T1+DWI+rsfMRI.connectivity,random=~1|Target,data=df)
#anova(m1)

#m1 <- lme(Balanced.accuracy~(Psychology+Constitutional+Serology+Disease+T1+DWI+rsfMRI.connectivity),random=~1|Target,data=df)
#summary(m1)
##anova(m1)

#m1 <- lme(Balanced.accuracy~(Metadata+T1+DWI+rsfMRI.connectivity),random=~1|Target,data=df)
#anova(m1)

#m1 <- lme(Balanced.accuracy~(Metadata+T1*DWI*rsfMRI.connectivity),random=~1|Target,data=df)
#anova(m1)


#constitutional inputs
df_constitutional = filter(df, Domain == "Constitutional")
df_constitutional_continuous = df_constitutional %>% filter(r2>0)
df_constitutional_categorical = df_constitutional %>% filter(r2==0)

m2 <- lme(Balanced.accuracy~(T1+DWI+rsfMRI.connectivity),random=~1|Target,data=df_constitutional)
summary(m2)
tab_model(m2)
tab_model(m2,digits=5,digits.p=5,file = "/Users/jruffle/Library/CloudStorage/OneDrive-UniversityCollegeLondon/MSc/MScProject/FIGURE_FILES/lmer_constitutional.html")


#m2.5 <- lme(Balanced.accuracy~(T1+DWI+rsfMRI.connectivity),random=~1|Target,data=df_constitutional_categorical)
#summary(m2.5)
#tab_model(m2.5,digits=5,digits.p=5,file = "/Users/jruffle/Library/CloudStorage/OneDrive-UniversityCollegeLondon/MSc/MScProject/FIGURE_FILES/lmer_constitutional_categorical.html")

#m2.55 <- lme(r2~(T1+DWI+rsfMRI.connectivity),random=~1|Target,data=df_constitutional_continuous)
#summary(m2.55)
#tab_model(m2.55,digits=5,digits.p=5,file = "/Users/jruffle/Library/CloudStorage/OneDrive-UniversityCollegeLondon/MSc/MScProject/FIGURE_FILES/lmer_constitutional_continuous.html")


#library(lme4)
#LME_02<-lmer(Balanced.accuracy~T1*DWI*rsfMRI.connectivity+(1|Target),
#            na.action=na.exclude,data = df_constitutional)
#summary(LME_02)
#anova(LME_02)
#sjPlot::plot_model(LME_02)

#summary(LME_02)
#anova(LME_02)

#m3 <- lme(Balanced.accuracy~(T1*DWI+rsfMRI.connectivity),random=~1|Target,data=df_constitutional_categorical)
#summary(m3)

#m4 <- lme(r2~(T1*DWI+rsfMRI.connectivity),random=~1|Target,data=df_constitutional_continuous)
#summary(m4)


#psychology inputs
df_psychology = filter(df, Domain == "Psychology")
df_psychology_continuous = df_psychology %>% filter(r2>0)
df_psychology_categorical = df_psychology %>% filter(r2==0)

#first find the fit with lme4lmer with backward feature selection
#m5 = lme4::lmer(Balanced.accuracy~Serology*Disease*T1*DWI*rsfMRI.connectivity+(1|Target),data=df_psychology)
#m5 = stepcAIC(m5, data=df_psychology, trace=TRUE, direction="backward")

m5 <- lme(Balanced.accuracy~(Serology+Disease+T1+DWI+rsfMRI.connectivity),random=~1|Target,data=df_psychology)
tab_model(m5)
sjPlot::plot_model(m5)
#m5 <- lme(Balanced.accuracy~(Serology+Disease+T1*DWI+rsfMRI.connectivity)|Target,data=df_psychology)

summary(m5)
tab_model(m5,digits=5,digits.p=5,file = "/Users/jruffle/Library/CloudStorage/OneDrive-UniversityCollegeLondon/MSc/MScProject/FIGURE_FILES/lmer_psychology.html")

#m6 <- lme(Balanced.accuracy~(Serology+Disease+T1+DWI+rsfMRI.connectivity),random=~1|Target,data=df_psychology_categorical)
#summary(m6)

#m7 <- lme(r2~(Serology+Disease+T1+DWI+rsfMRI.connectivity),random=~1|Target,data=df_psychology_continuous)
#summary(m7)


#disease inputs
df_disease = filter(df, Domain == "Disease")
df_disease_continuous = df_disease %>% filter(r2>0)
df_disease_categorical = df_disease %>% filter(r2==0)

m8 <- lme(Balanced.accuracy~(Serology+Psychology+T1+DWI+rsfMRI.connectivity),random=~1|Target,data=df_disease)
summary(m8)
tab_model(m8)
tab_model(m8,digits=5,digits.p=5,file = "/Users/jruffle/Library/CloudStorage/OneDrive-UniversityCollegeLondon/MSc/MScProject/FIGURE_FILES/lmer_disease.html")


#serology inputs
df_serology = filter(df, Domain == "Serology")
df_serology_continuous = df_serology %>% filter(r2>0)
df_serology_categorical = df_serology %>% filter(r2==0)

m9 <- lme(Balanced.accuracy~(Disease+Psychology+T1*DWI+rsfMRI.connectivity),random=~1|Target,data=df_serology)
summary(m9)
tab_model(m9,digits=5,digits.p=5,file = "/Users/jruffle/Library/CloudStorage/OneDrive-UniversityCollegeLondon/MSc/MScProject/FIGURE_FILES/lmer_serology.html")

m10 <- lme(r2~(Disease+Psychology+T1+DWI+rsfMRI.connectivity),random=~1|Target,data=df_serology_continuous)
summary(m10)
tab_model(m10)
tab_model(m10,digits=5,digits.p=5,file = "/Users/jruffle/Library/CloudStorage/OneDrive-UniversityCollegeLondon/MSc/MScProject/FIGURE_FILES/lmer_serology_continuous.html")


m11 <- lme(r2~(Disease+Psychology+T1*DWI+rsfMRI.connectivity),random=~1|Target,data=df_serology_continuous)
summary(m11)
