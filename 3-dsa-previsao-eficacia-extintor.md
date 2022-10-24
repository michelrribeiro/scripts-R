Projeto DSA: Previsão da Eficácia de Extintores
================
Michel Ribeiro

### Projeto: desenvolver modelo de classificação para prever se um extintor extinguirá o fogo baseado em parâmetros de teste.

#### O dataset utilizado está disponível em: [dataset](https://www.muratkoklu.com/datasets/vtdhnd07.php)

Definindo a pasta de trabalho:

``` r
setwd('~/Desktop/DSA/big-Data-R-Azure/projeto_final_02.v02')
```

Carregando os pacotes:

``` r
library(dplyr, quietly = T)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(ggplot2, quietly = T)
library(caret, quietly = T)
library(caretEnsemble, quietly = T)
```

    ## 
    ## Attaching package: 'caretEnsemble'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     autoplot

``` r
library(CatEncoders)
```

    ## 
    ## Attaching package: 'CatEncoders'

    ## The following object is masked from 'package:base':
    ## 
    ##     transform

``` r
source("~/Desktop/DSA/big-Data-R-Azure/0-functions/functions.R")
```

Carregando os dados e visualizando:

``` r
library(readxl, quietly = T)
df <- read_xlsx('./data/Acoustic_Extinguisher_Fire_Dataset.xlsx') %>% as.data.frame
detach(package: readxl)

head(df)
```

    ##   SIZE     FUEL DISTANCE DESIBEL AIRFLOW FREQUENCY STATUS
    ## 1    1 gasoline       10      96     0.0        75      0
    ## 2    1 gasoline       10      96     0.0        72      1
    ## 3    1 gasoline       10      96     2.6        70      1
    ## 4    1 gasoline       10      96     3.2        68      1
    ## 5    1 gasoline       10     109     4.5        67      1
    ## 6    1 gasoline       10     109     7.8        66      1

``` r
glimpse(df)
```

    ## Rows: 17,442
    ## Columns: 7
    ## $ SIZE      <dbl> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, …
    ## $ FUEL      <chr> "gasoline", "gasoline", "gasoline", "gasoline", "gasoline", …
    ## $ DISTANCE  <dbl> 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, …
    ## $ DESIBEL   <dbl> 96, 96, 96, 96, 109, 109, 103, 95, 102, 93, 93, 95, 110, 111…
    ## $ AIRFLOW   <dbl> 0.0, 0.0, 2.6, 3.2, 4.5, 7.8, 9.7, 12.0, 13.3, 15.4, 15.1, 1…
    ## $ FREQUENCY <dbl> 75, 72, 70, 68, 67, 66, 65, 60, 55, 52, 51, 50, 48, 47, 46, …
    ## $ STATUS    <dbl> 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, …

``` r
colSums(is.na(df))
```

    ##      SIZE      FUEL  DISTANCE   DESIBEL   AIRFLOW FREQUENCY    STATUS 
    ##         0         0         0         0         0         0         0

Visualizando os valores únicos para cada variável categórica:

``` r
col_cat <- c("SIZE", "FUEL", "STATUS")
col_num <- colnames(df[!colnames(df) %in% col_cat])

for (n in col_cat){print(unique(df[, n]))}
```

    ## [1] 1 2 3 4 5 6 7
    ## [1] "gasoline" "thinner"  "kerosene" "lpg"     
    ## [1] 0 1

Avaliando o equilíbrio de classes:

``` r
paste("Classe 1:", sum(df$STATUS), "|", 
      "Classe 0:", (length(df$STATUS) - sum(df$STATUS)))
```

    ## [1] "Classe 1: 8683 | Classe 0: 8759"

As classes estão bem equilibradas.

Avaliando as variáveis numéricas quanto às suas distribuições e seus
coeficientes de assimetria e de curtose:

``` r
par(mfrow = c(2, 2), mar = rep(2, 4))

for (n in 1:length(col_num)){
  df[ , col_num[n]] %>% hist(main = col_num[n])
}
```

![](dsa-previsao-eficacia-extintor_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
for (n in 1:length(col_num)){
  df[ , col_num[n]] %>% boxplot(main = col_num[n])
  }
```

![](dsa-previsao-eficacia-extintor_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
require(timeDate)
```

    ## Loading required package: timeDate

``` r
for (n in 1:length(col_num)){
  print(paste0(col_num[n], 
              " - Assimetria: ", round(skewness(df[ , col_num[n]]), 3),
              ", Curtose: ", round(kurtosis(df[ , col_num[n]]), 3)))
}
```

    ## [1] "DISTANCE - Assimetria: 0, Curtose: -1.207"
    ## [1] "DESIBEL - Assimetria: -0.179, Curtose: -0.577"
    ## [1] "AIRFLOW - Assimetria: 0.244, Curtose: -1.198"
    ## [1] "FREQUENCY - Assimetria: 0.435, Curtose: -0.901"

``` r
detach(package: timeDate)
```

Os gráficos e os valores mostram que, no geral, as variáveis não possuem
distribuição normal. Também indicam uma distribuição bimodal para
“DESIBEL”. Os dados serão normalizados na sequência do processo.

A variável “FUEL” passará por transformação com one hot encoding e, em
seguida, as três variáveis categóricas serão convertidas para fator:

``` r
oneHotEnc <- transform(OneHotEncoder.fit(X = data.matrix(df$FUEL)),
                       data.matrix(df$FUEL), sparse = F) %>% as.data.frame

df2 <- cbind(df, oneHotEnc) 

head(df2)
```

    ##   SIZE     FUEL DISTANCE DESIBEL AIRFLOW FREQUENCY STATUS V1 V2 V3 V4
    ## 1    1 gasoline       10      96     0.0        75      0  1  0  0  0
    ## 2    1 gasoline       10      96     0.0        72      1  1  0  0  0
    ## 3    1 gasoline       10      96     2.6        70      1  1  0  0  0
    ## 4    1 gasoline       10      96     3.2        68      1  1  0  0  0
    ## 5    1 gasoline       10     109     4.5        67      1  1  0  0  0
    ## 6    1 gasoline       10     109     7.8        66      1  1  0  0  0

``` r
head(df2[df2$V2 == 1, ], 1); head(df2[df2$V3 == 1, ], 1); head(df2[df2$V4 == 1, ], 1)
```

    ##       SIZE     FUEL DISTANCE DESIBEL AIRFLOW FREQUENCY STATUS V1 V2 V3 V4
    ## 10261    1 kerosene       10      96       0        75      0  0  1  0  0

    ##       SIZE FUEL DISTANCE DESIBEL AIRFLOW FREQUENCY STATUS V1 V2 V3 V4
    ## 15391    6  lpg       10      96       0        75      0  0  0  1  0

    ##      SIZE    FUEL DISTANCE DESIBEL AIRFLOW FREQUENCY STATUS V1 V2 V3 V4
    ## 5131    1 thinner       10      96       0        75      0  0  0  0  1

``` r
colnames(df2)[8:11] <- c("Gasoline", "Kerosene", "LPG", "Thinner")
df2 <- df2[ , !colnames(df2) %in% c("FUEL")]

col_cat <- col_cat[c(1, 3)] %>% append(colnames(df2)[7:10])

df2 <- df_to_factor(df2, col_cat)
```

As variáveis numéricas serão normalizadas por MinMaxScaler:

``` r
df2 <- scale_data_frame(df2, col_num)

glimpse(df2)
```

    ## Rows: 17,442
    ## Columns: 10
    ## $ SIZE      <fct> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, …
    ## $ DISTANCE  <dbl> -1.643121, -1.643121, -1.643121, -1.643121, -1.643121, -1.64…
    ## $ DESIBEL   <dbl> -0.0464402, -0.0464402, -0.0464402, -0.0464402, 1.5458977, 1…
    ## $ AIRFLOW   <dbl> -1.4728430, -1.4728430, -0.9238761, -0.7971915, -0.5227080, …
    ## $ FREQUENCY <dbl> 2.07214190, 1.92886961, 1.83335475, 1.73783988, 1.69008245, …
    ## $ STATUS    <fct> 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, …
    ## $ Gasoline  <fct> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, …
    ## $ Kerosene  <fct> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    ## $ LPG       <fct> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    ## $ Thinner   <fct> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …

Divindo os dados em treino e teste a partir de 600 registros para
escolha do modelo:

``` r
set.seed(2022)
df2_sample <- rbind(df2[df2$STATUS == 0, ]
                    [sample(1:nrow(df2[df2$STATUS == 0, ]), size = 300), ], 
                    df2[df2$STATUS == 1, ]
                    [sample(1:nrow(df2[df2$STATUS == 1, ]), size = 300), ])

idx <- sample(1:nrow(df2_sample), size = 0.75 * nrow(df2_sample))
x_train <- df2_sample[idx, !colnames(df2_sample) %in% c("STATUS")]
x_test <- df2_sample[-idx, !colnames(df2_sample) %in% c("STATUS")]
y_train <- df2_sample[idx, "STATUS"]
y_test <- df2_sample[-idx, "STATUS"]

rm(list = ls()[!ls() %in% c("df2", "x_train", "x_test", "y_train", "y_test")])
```

Criando e treinando os modelos:

``` r
fitControl= trainControl(method="cv", number = 5, 
                         savePredictions = "final", allowParallel = TRUE)

model_list <- caretList(x = x_train, y = y_train, trControl = fitControl, 
                        methodList = c("glm", "adaboost", "cforest", 
                                       "treebag", "ada"), 
                        tuneList = NULL, continue_on_fail = FALSE)
```

    ## Warning in trControlCheck(x = trControl, y = target): indexes not defined in
    ## trControl. Attempting to set them ourselves, so each model in the ensemble will
    ## have the same resampling indexes.

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

Realizando as predições:

``` r
pred_glm <- predict.train(model_list$glm, newdata = x_test)
```

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
    ## prediction from a rank-deficient fit may be misleading

``` r
pred_adaboost <- predict.train(model_list$adaboost, newdata = x_test)
pred_cforest <- predict.train(model_list$cforest, newdata = x_test)
pred_treebag <- predict.train(model_list$treebag, newdata = x_test)
pred_ada <- predict.train(model_list$ada, newdata = x_test)
```

Avaliando os modelos:

``` r
result_models <- resamples(list(GLM = model_list$glm, 
                                ADAB = model_list$adaboost, 
                                CFOR = model_list$cforest, 
                                TREEBAG = model_list$treebag, 
                                ADA = model_list$ada))

summary(result_models)
```

    ## 
    ## Call:
    ## summary.resamples(object = result_models)
    ## 
    ## Models: GLM, ADAB, CFOR, TREEBAG, ADA 
    ## Number of resamples: 5 
    ## 
    ## Accuracy 
    ##              Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## GLM     0.8539326 0.8777778 0.9000000 0.8954263 0.9120879 0.9333333    0
    ## ADAB    0.7977528 0.8461538 0.8888889 0.8776702 0.9222222 0.9333333    0
    ## CFOR    0.8089888 0.8555556 0.9010989 0.8775731 0.9111111 0.9111111    0
    ## TREEBAG 0.7977528 0.8222222 0.8777778 0.8531281 0.8777778 0.8901099    0
    ## ADA     0.8651685 0.8901099 0.9222222 0.9110557 0.9333333 0.9444444    0
    ## 
    ## Kappa 
    ##              Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## GLM     0.7081967 0.7553139 0.7996042 0.7907646 0.8242395 0.8664688    0
    ## ADAB    0.5962702 0.6919729 0.7768964 0.7551754 0.8441366 0.8666008    0
    ## CFOR    0.6195625 0.7108255 0.8018389 0.7552638 0.8217822 0.8223100    0
    ## TREEBAG 0.5968797 0.6435644 0.7548291 0.7062739 0.7557967 0.7802994    0
    ## ADA     0.7308468 0.7799807 0.8439822 0.8220335 0.8664688 0.8888889    0

``` r
dotplot(result_models)
```

![](dsa-previsao-eficacia-extintor_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

A partir dos valores da acurácia acima, o modelo “ada” foi escolhido
para ser treinado com o dataset completo.

``` r
rm(list = ls()[!ls() %in% c("df2", "fitControl")])

x_train <- df2[ , !colnames(df2) %in% c("STATUS")]
y_train <- df2[ , "STATUS"]

final_model <- train(x = x_train, y = y_train, method = 'ada', trControl = fitControl)
```

Testando novos dados:

``` r
df_test <- data.frame(SIZE = factor(c("1", "2", "2", "3", "6")),
                      DISTANCE = scale(c(40, 70, 80, 120, 190)),
                      DESIBEL = scale(c(30, 77, 63, 48, 105)),
                      AIRFLOW = scale(c(1.2, 2.2, 10.55, 2.4, 11.1)),
                      FREQUENCY = scale(c(5, 70, 43, 29, 5)),
                      Gasoline = factor(c("0", "1", "0", "0", "1")),
                      Kerosene = factor(c("1", "0", "0", "1", "0")),
                      LPG = factor(c("0", "0", "1", "0", "0")),
                      Thinner = factor(c("0", "0", "0", "0", "0")))

df_results <- cbind(df_test, predict.train(final_model, newdata = df_test))                      

df_results
```

    ##   SIZE   DISTANCE     DESIBEL    AIRFLOW   FREQUENCY Gasoline Kerosene LPG
    ## 1    1 -1.0366421 -1.21176214 -0.8763684 -0.92452311        0        1   0
    ## 2    2 -0.5183211  0.43427314 -0.6720867  1.44138248        1        0   0
    ## 3    2 -0.3455474 -0.05603524  1.0336653  0.45862170        0        0   1
    ## 4    3  0.3455474 -0.58136565 -0.6312304 -0.05095797        0        1   0
    ## 5    6  1.5549632  1.41488989  1.1460202 -0.92452311        1        0   0
    ##   Thinner predict.train(final_model, newdata = df_test)
    ## 1       0                                             1
    ## 2       0                                             0
    ## 3       0                                             1
    ## 4       0                                             0
    ## 5       0                                             1
