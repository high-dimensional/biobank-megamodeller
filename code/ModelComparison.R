library(tidyverse)
library(MASS)
library(nlme)
library(cAIC4)
library(sjPlot)

df = read.csv('/Users/jruffle/Library/CloudStorage/OneDrive-UniversityCollegeLondon/MSc/MScProject/FIGURE_FILES/metrics_comparison_test.csv')
head(df)

#constitutional inputs
df_constitutional = filter(df, Domain == "Constitutional")
df_constitutional_continuous = df_constitutional %>% filter(r2>0)
df_constitutional_categorical = df_constitutional %>% filter(r2==0)

m2 <- lme(Balanced.accuracy~(T1+DWI+rsfMRI.connectivity),random=~1|Target,data=df_constitutional)
summary(m2)
tab_model(m2)
tab_model(m2,digits=5,digits.p=5,file = "/Users/jruffle/Library/CloudStorage/OneDrive-UniversityCollegeLondon/MSc/MScProject/FIGURE_FILES/lmer_constitutional.html")

#psychology inputs
df_psychology = filter(df, Domain == "Psychology")
df_psychology_continuous = df_psychology %>% filter(r2>0)
df_psychology_categorical = df_psychology %>% filter(r2==0)

m5 <- lme(Balanced.accuracy~(Serology+Disease+T1+DWI+rsfMRI.connectivity),random=~1|Target,data=df_psychology)
tab_model(m5)
sjPlot::plot_model(m5)

summary(m5)
tab_model(m5,digits=5,digits.p=5,file = "/Users/jruffle/Library/CloudStorage/OneDrive-UniversityCollegeLondon/MSc/MScProject/FIGURE_FILES/lmer_psychology.html")

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
