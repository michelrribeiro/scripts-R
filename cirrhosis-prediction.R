options(warn = -1)
path = "/home/michelrribeiro/Desktop/DSA/big-Data-R-Azure/proj-cirrhosis-prediction"
setwd(path)

# Carregando pacotes:
library(dplyr, quietly = T)
source("~/Desktop/DSA/big-Data-R-Azure/0-functions/functions.R")


# O projeto trata da previsão de ocorrência ou não de cirrose a partir de uma
# base de dados com vários fatores a serem considerados.

# Carga da base de dados e do dicionário de dados.
df <- read.csv(paste0(path, "/cirrhosis.csv"))
data.dict <- read.csv(paste0(path, "/data.dict.csv"), sep = ";")

#View(data.dict)

# Análise exploratória e transformação de dados:
glimpse(df)

cat_vars <- c("Status", "Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema")
num_vars <- colnames(df)[!colnames(df) %in% cat_vars]

colSums(is.na(df))

count_unique_df(df[ , cat_vars])

for( n in 1:length(cat_vars)){
  print(cat_vars[n])
  print(table(df[ , cat_vars[n]]))
  }

# As variáveis categóricas serão tratadas com a remoção dos valores NA em um novo dataframe e, 
# em seguida, a aplicação de "Label Encoding" para "Edema":
#  No edema, no therapy (N) < edema without terapy or solved by it (S) < edema despite therapy (Y)
#
# A variável "Status" será removida, pois  os status "Censored", "Censored Due Tx Liver" e 
# "Death" apresentam poucos resultados claros e não parecem ter uma confiabilidade de informação.
#
# Além disso, as outras variáveis categóricas serão transformadas em valores binários:

df2 <- df[!is.na(df$Ascites), ]

colSums(is.na(df2))

df2 <- df2[ , !colnames(df2) %in% "Status"]

df2$Edema <- ifelse(df2$Edema == "N", 1, 
                    ifelse(df2$Edema == "S", 2, 3))

df2$Drug <- ifelse(df2$Drug == "Placebo", 0, 1)

df2$Sex <- ifelse(df2$Sex == "M", 0, 1)
colnames(df2)[colnames(df2) %in% "Sex"] <- "Female"

df2$Ascites <- ifelse(df2$Ascites == "N", 0, 1)

df2$Hepatomegaly <- ifelse(df2$Hepatomegaly == "N", 0, 1)

df2$Spiders <- ifelse(df2$Spiders == "N", 0, 1)

rm(list = ls()[!ls() %in% c("data.dict", "df", "df2", "df_to_factor", 
                            "fill_na_median", "num_vars", "scale_data_frame")])

# As variáveis numéricas possuem valores NA, que serão alterados 
# para o valor da mediana de cada variável. As variáveis também serão normalizadas:
summary(df2[ , num_vars])
df2[, num_vars] <- sapply(df2[, num_vars], fill_na_median)
df2 <- scale_data_frame(df2, num_vars[!num_vars %in% "Stage"])

# A variável alvo é a que detalha o estágio dos danos no fígado, sendo o estágio 4
# a cirrose e o estágio 3 o início da fibrose. 3 e 4 serão considerados como o valor
# para doença (0):
df2$Stage <- ifelse(df2$Stage > 2, 0, 1)

# Transformando as variáveis binárias para o tipo fator:
df2 <- df_to_factor(df = df2, 
                    col_list = append(colnames(df2)[!colnames(df2) %in% num_vars], "Stage"))

# Balanceamento de classes na variável alvo e separação dos dados restantes em outro
# data frame:
set.seed(123)
table(df2$Stage)

df2_sample <- rbind(df2[df2$Stage == 1, ]
                    [sample(1:nrow(df2[df2$Stage == 1, ]), size = 83), ], 
                    df2[df2$Stage == 0, ])

# Dividindo os dados em treino e teste:
set.seed(123)
idx <- sample(1:nrow(df2_sample), size = 0.8 * nrow(df2_sample))
col_names <- colnames(df2_sample)[!colnames(df2_sample) %in% "Stage"]

x_train <- df2_sample[idx, col_names]
x_test <- df2_sample[-idx, col_names]
y_train <- df2_sample[idx, !colnames(df2_sample) %in% col_names]
y_test <- df2_sample[-idx, !colnames(df2_sample) %in% col_names]

# Criando e treinando os modelos:
library(caret, quietly = T)
library(caretEnsemble, quietly = T)

fitControl <- trainControl(method="repeatedcv", 
                           number = 5, 
                           repeats = 5,
                           savePredictions = "final", 
                         allowParallel = TRUE)

model_list <- caretList(x = x_train, y = y_train, trControl = fitControl, 
                        methodList = c("glm", "adaboost", "cforest", 
                                       "treebag", "ada"), 
                        tuneList = NULL, continue_on_fail = FALSE)

# Realizando as predições:
pred_glm <- predict.train(model_list$glm, newdata = x_test)
pred_adaboost <- predict.train(model_list$adaboost, newdata = x_test)
pred_cforest <- predict.train(model_list$cforest, newdata = x_test)
pred_treebag <- predict.train(model_list$treebag, newdata = x_test)
pred_ada <- predict.train(model_list$ada, newdata = x_test)

# Avaliando os modelos:
result_models <- resamples(list(GLM = model_list$glm, 
                                ADAB = model_list$adaboost, 
                                CFOR = model_list$cforest, 
                                TREEBAG = model_list$treebag, 
                                ADA = model_list$ada))

rm(list = ls()[!ls() %in% c("df2", "result_models", "fitControl", "model_list", "x_test", "y_test")])

# Comparação entre os modelos:
summary(result_models)$statistics$Accuracy

# O modelo com maior acurácia média foi o Conditional Inference Random Forest:

modelo.final <- train(x = df2[ , !colnames(df2) %in% "Stage"], 
                      y = df2[ , "Stage"],
                      trControl = fitControl, 
                      method = "cforest")

modelo.final