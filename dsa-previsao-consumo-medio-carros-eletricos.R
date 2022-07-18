# Definindo diretório de trabalho:
setwd("~/Desktop/DSA/big-Data-R-Azure/projeto_final_01.v02")

# Desativando os avisos para um console mais limpo:
# options(warn = -1)

# Carregando pacotes:
library(dplyr, quietly = T)
library(caret, quietly = T)
library(caretEnsemble, quietly = T)
source("~/Desktop/DSA/big-Data-R-Azure/0-functions/functions.R")

# Projeto final 01 - segunda proposta:
# Desenvolver modelo de machine learning para prever o consumo médio de carros elétricos.

# O dataset utilizado está disponível em:
# https://data.mendeley.com/datasets/tb9yrptydn/2

library(readxl, quietly = T)
df <- readxl::read_xlsx('./FEV-data-Excel.xlsx') %>% as.data.frame
detach(package: readxl)

# Visualizando subset com linhas que possuem valores em branco:
glimpse(df)
colSums(is.na(df))[(colSums(is.na(df)) > 0)]
View(df[rowSums(is.na(df)) > 0, ])

# Criando subset sem os valores em branco:
df2 <- df[rowSums(is.na(df)) == 0, ]

# Renomeando colunas para remover espaços e evitar problemas nos modelos:
colnames(df2) <- gsub(" ", "", colnames(df2))
colnames(df2)

# Criando subset sem valores em branco e com variáveis numéricas apenas:
col_to_drop <- c("Carfullname", "Make", "Model", "Typeofbrakes", "Drivetype")
df2_num <- df2[ , !colnames(df2) %in% col_to_drop]

dim(df2_num)
summary(df2_num)

# Avaliando correlação:
library(reshape2, quietly = T)
corr_matrix <- data.frame(melt(cor(df2_num)), stringsAsFactors = F)
detach(package: reshape2)

# Correlação das variáveis preditoras entre si:
View(corr_matrix[corr_matrix$Var2 != "mean-Energyconsumption[kWh/100km]" & 
                   corr_matrix$Var1 != "mean-Energyconsumption[kWh/10km]" &
                   corr_matrix$Var1 != corr_matrix$Var2, ])

# A variável "Permissable gross weight [kg]" tem forte correlação com a variável 
# Minimal empty weight [kg]" e será removida.

# A variável "Engine power [KM]" tem correlação com as variáveis "Maximum torque [Nm]"
# e "Maximum speed [kph]", sendo que a primeira influencia nas outras e apenas  
# "Engine power [KM]" será mantida.

# Correlação das variáveis preditoras com a variável alvo:
View(corr_matrix[corr_matrix$Var1 != "mean-Energyconsumption[kWh/100km]" & 
                   corr_matrix$Var2 == "mean-Energyconsumption[kWh/100km]", ])

# Serão consideradas para o modelo as variáveis que possuem 
# correlação acima de 0.5 ou abaixo de -0.5 com a variável alvo.

# As variáveis categóricas serão abandonadas, pois a diferença de marca
# também pode ser sentida na diferença de preço para parâmetros similares.

val_to_drop <- c("Minimalemptyweight[kg]", "Maximumtorque[Nm]", 
                 "Maximumspeed[kph]")

var_selec <- corr_matrix[((corr_matrix$value > 0.5 | corr_matrix$value < -0.5) & 
                           (corr_matrix$Var2 == "mean-Energyconsumption[kWh/100km]") &
                            !corr_matrix$Var1 %in% val_to_drop), 1] %>% as.character

# Aplicando normalização no dataset com as variáveis selecionadas:
df2_num_scale <- scale_data_frame(df2_num[, var_selec])

# Dividindo em dados de treino e teste na proporção 75/25:
set.seed(1234)

indexes <- sample(1:nrow(df2_num_scale), size = 0.75 * nrow(df2_num_scale))
x_train <- df2_num_scale[indexes, 1:(length(df2_num_scale)-1)]
y_train <- df2_num_scale[indexes, length(df2_num_scale)]
x_test <- df2_num_scale[-indexes, 1:(length(df2_num_scale)-1)]
y_test <- df2_num_scale[-indexes, length(df2_num_scale)]

rm(col_to_drop, corr_matrix, df, df2,  df2_num, df2_num_scale, indexes, val_to_drop, var_selec)

# Treinando os modelos escolhidos com caretEnsemble:
fitControl= trainControl(method="cv", number = 5, 
                         savePredictions = "final", allowParallel = TRUE)

model_list <- caretList(x = x_train, y = y_train, trControl = fitControl, 
                        methodList = c("lm", "svmRadial", "rf", "bridge", "xgbLinear"), 
                        tuneList = NULL, continue_on_fail = FALSE)

# Fazendo as predições:
pred_lm <- predict.train(model_list$lm, newdata = x_test)
pred_svm <- predict.train(model_list$svmRadial, newdata = x_test)
pred_rf <- predict.train(model_list$rf, newdata = x_test)
pred_bridge <- predict.train(model_list$bridge, newdata = x_test)
pred_xgbL <- predict.train(model_list$xgbLinear, newdata = x_test)

# Avaliando os modelos:
 result_models <- 
   data.frame(LM = c(min(model_list$lm$results$RMSE), RMSE(pred_lm, y_test), cor(pred_lm, y_test)), 
              SVM = c(min(model_list$svmRadial$results$RMSE), RMSE(pred_svm, y_test), cor(pred_svm, y_test)), 
              RF = c(min(model_list$rf$results$RMSE), RMSE(pred_rf, y_test), cor(pred_rf, y_test)), 
              BRIDGE = c(min(model_list$bridge$results$RMSE), RMSE(pred_bridge, y_test), cor(pred_bridge, y_test)), 
              XGBL = c(min(model_list$xgbLinear$results$RMSE), RMSE(pred_xgbL, y_test), cor(pred_xgbL, y_test)))

rownames(result_models) <- c("RMSE_model", "RMSE_predict", "Corr_Pred_Real")

result_models

# Conclusão: O modelo Random Forest obteve o melhor desempenho e foi o escolhido para este caso.

modelo_final <- model_list$rf
