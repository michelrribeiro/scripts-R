# Definindo a pasta de trabalho:
setwd('~/Desktop/DSA/big-Data-R-Azure/projeto_final_01')

# Carregando pacotes:
library(dplyr)
library(ggplot2)
library(data.table)
library(caret)

# Projeto final 01:
# Desenvolver modelo de machine learning para prever se um clique em um anúncio resultará em um download.

# O dataset utilizado está disponível em:
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

# O dicionário de dados foi copiado e salvo como um arquivo .json para ficar disponível na sessão:
library(jsonlite)
data_dict <- jsonlite::read_json('./data_fields.json')
detach(package: jsonlite)
View(data_dict)

# Como o dataset de dados de treino possui muitos registros, o trabalho inicial será feito
# em cima da amostra randomizada que já é disponibilizada  pelo site:
dt <- fread('./data/train_sample.csv') # %>% as.data.frame
dim(dt)
str(dt)
View(dt)

# Calculando a quantidade de valores NA e valores únicos:
colSums(is.na(dt))

for (n in c(1:8)){
  colnames(dt)[n] %>% paste(unique(dt[ , ..n]) %>% count) %>% print
  }

# Como o horário específico de download possui muitos valores ausentes, a coluna será descartada 
# A coluna click_time será transformada em uma variável categórica ordinária: 0, 1, 2, 3 para cada
# intervalo de 6h a partir da meia-noite.
# Também será removida a variável "ip", pois não será usada na criação do modelo.
factor_click_time <- function(number){
  if (number <= 6){0
    } else if (number <= 12){1
    } else if (number <= 18){2
    } else {3}
  }

dt$click_time_fac <- as.integer(format(as.POSIXct(dt$click_time), format = "%H"))
dt$click_time_fac <- sapply(dt$click_time_fac, factor_click_time)

dt <- dt[ , !c("ip", "click_time","attributed_time")]

# Alterando as variáveis para o tipo fator:
fac_cols <- c(colnames(dt))
dt <- dt[, (fac_cols) := lapply(.SD, as.factor), .SDcols = fac_cols]

str(dt)
summary(dt)

# Separação em amostra de treino e amostra de teste:
set.seed(16)
idx <- sample(c(rep(0, 0.7 * nrow(dt)), rep(1, 0.3 * nrow(dt))))

train.dt <- dt[which(idx == 0), ]
test.dt <- dt[which(idx == 1), ]

summary(train.dt$is_attributed)

# Balanceando o dataset de treino e diminuindo o volume de dados para conseguir processar e,
# em seguida, removendo variáveis para liberar memória.
set.seed(1604)
train.dt2 <- rbind(subset(train.dt, is_attributed == 1), 
             sample_n(subset(train.dt, is_attributed == 0), size = 159))

summary(train.dt2$is_attributed)
rm(dt, fac_cols, factor_click_time, idx, n, train.dt)

# Criando o modelo:
# Boosted Classification Trees:
#install.packages('ada')
library(ada)
ada.model <- train(x = train.dt2[, c(1:4, 6)], y = train.dt2$is_attributed, 
                   method = 'ada')

# Aplicando o ada.model aos dados de teste:
ada.predict <- predict(ada.model, test.dt, type = "raw")

confusionMatrix(table(data = ada.predict, reference = test.dt$is_attributed), positive = '1')

# Acurácia: 92%