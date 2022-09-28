Projeto: Previsão do Consumo Médio de Carros Elétricos
================
Michel Ribeiro

### Projeto: Criar um modelo de classificação para a previsão de ocorrência ou não de cirrose a partir de uma base de dados com vários atributos.

#### O dataset utilizado está disponível em: [dataset](https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset)

Definindo a pasta de trabalho:

``` r
path = "/home/michelrribeiro/Desktop/DSA/big-Data-R-Azure/proj-cirrhosis-prediction"
setwd(path)
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
library(caret, quietly = T)
library(caretEnsemble, quietly = T)
```

    ## 
    ## Attaching package: 'caretEnsemble'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     autoplot

``` r
library(xgboost, quietly = T)
```

    ## 
    ## Attaching package: 'xgboost'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     slice

``` r
source("~/Desktop/DSA/big-Data-R-Azure/0-functions/functions.R")
```

Carregando o dataset:

``` r
df <- read.csv(paste0(path, "/cirrhosis.csv"))
```

O dicionário de dados foi copiado para um arquivo .csv e carregado na
sessão:

``` r
data.dict <- read.csv(paste0(path, "/data.dict.csv"), sep = ";")
data.dict %>% as.data.frame
```

    ##          Feature
    ## 1             ID
    ## 2         N_Days
    ## 3         Status
    ## 4           Drug
    ## 5            Age
    ## 6            Sex
    ## 7        Ascites
    ## 8   Hepatomegaly
    ## 9        Spiders
    ## 10         Edema
    ## 11     Bilirubin
    ## 12   Cholesterol
    ## 13       Albumin
    ## 14        Copper
    ## 15      Alk_Phos
    ## 16          SGOT
    ## 17 Triglycerides
    ## 18     Platelets
    ## 19   Prothrombin
    ## 20         Stage
    ##                                                                                                                                                                     Description
    ## 1                                                                                                                                                             unique identifier
    ## 2                                                            number of days between registration and the earlier of death, transplantation, or study analysis time in July 1986
    ## 3                                                                                               status of the patient C (censored), CL (censored due to liver tx), or D (death)
    ## 4                                                                                                                                       type of drug D-penicillamine or placebo
    ## 5                                                                                                                                                                 age in [days]
    ## 6                                                                                                                                                        M (male) or F (female)
    ## 7                                                                                                                                         presence of ascites N (No) or Y (Yes)
    ## 8                                                                                                                                    presence of hepatomegaly N (No) or Y (Yes)
    ## 9                                                                                                                                         presence of spiders N (No) or Y (Yes)
    ## 10 presence of edema N (no edema and no diuretic therapy for edema), S (edema present without diuretics, or edema resolved by diuretics), or Y (edema despite diuretic therapy)
    ## 11                                                                                                                                                   serum bilirubin in [mg/dl]
    ## 12                                                                                                                                                 serum cholesterol in [mg/dl]
    ## 13                                                                                                                                                           albumin in [gm/dl]
    ## 14                                                                                                                                                     urine copper in [ug/day]
    ## 15                                                                                                                                            alkaline phosphatase in [U/liter]
    ## 16                                                                                                                                                               SGOT in [U/ml]
    ## 17                                                                                                                                                     triglicerides in [mg/dl]
    ## 18                                                                                                                                                platelets per cubic [ml/1000]
    ## 19                                                                                                                                              prothrombin time in seconds [s]
    ## 20                                                                                                                                  histologic stage of disease (1, 2, 3, or 4)

Visualização dos dados:

``` r
glimpse(df)
```

    ## Rows: 418
    ## Columns: 20
    ## $ ID            <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1…
    ## $ N_Days        <int> 400, 4500, 1012, 1925, 1504, 2503, 1832, 2466, 2400, 51,…
    ## $ Status        <chr> "D", "C", "D", "D", "CL", "D", "C", "D", "D", "D", "D", …
    ## $ Drug          <chr> "D-penicillamine", "D-penicillamine", "D-penicillamine",…
    ## $ Age           <int> 21464, 20617, 25594, 19994, 13918, 24201, 20284, 19379, …
    ## $ Sex           <chr> "F", "F", "M", "F", "F", "F", "F", "F", "F", "F", "F", "…
    ## $ Ascites       <chr> "Y", "N", "N", "N", "N", "N", "N", "N", "N", "Y", "N", "…
    ## $ Hepatomegaly  <chr> "Y", "Y", "N", "Y", "Y", "Y", "Y", "N", "N", "N", "Y", "…
    ## $ Spiders       <chr> "Y", "Y", "N", "Y", "Y", "N", "N", "N", "Y", "Y", "Y", "…
    ## $ Edema         <chr> "Y", "N", "S", "S", "N", "N", "N", "N", "N", "Y", "N", "…
    ## $ Bilirubin     <dbl> 14.5, 1.1, 1.4, 1.8, 3.4, 0.8, 1.0, 0.3, 3.2, 12.6, 1.4,…
    ## $ Cholesterol   <int> 261, 302, 176, 244, 279, 248, 322, 280, 562, 200, 259, 2…
    ## $ Albumin       <dbl> 2.60, 4.14, 3.48, 2.54, 3.53, 3.98, 4.09, 4.00, 3.08, 2.…
    ## $ Copper        <int> 156, 54, 210, 64, 143, 50, 52, 52, 79, 140, 46, 94, 40, …
    ## $ Alk_Phos      <dbl> 1718.0, 7394.8, 516.0, 6121.8, 671.0, 944.0, 824.0, 4651…
    ## $ SGOT          <dbl> 137.95, 113.52, 96.10, 60.63, 113.15, 93.00, 60.45, 28.3…
    ## $ Tryglicerides <int> 172, 88, 55, 92, 72, 63, 213, 189, 88, 143, 79, 95, 130,…
    ## $ Platelets     <int> 190, 221, 151, 183, 136, NA, 204, 373, 251, 302, 258, 71…
    ## $ Prothrombin   <dbl> 12.2, 10.6, 12.0, 10.3, 10.9, 11.0, 9.7, 11.0, 11.0, 11.…
    ## $ Stage         <int> 4, 3, 4, 4, 3, 3, 3, 3, 2, 4, 4, 4, 3, 4, 3, 3, 4, 4, 3,…

``` r
cat_vars <- c("Status", "Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", 
              "Edema")
num_vars <- colnames(df)[!colnames(df) %in% cat_vars]

colSums(is.na(df))
```

    ##            ID        N_Days        Status          Drug           Age 
    ##             0             0             0           106             0 
    ##           Sex       Ascites  Hepatomegaly       Spiders         Edema 
    ##             0           106           106           106             0 
    ##     Bilirubin   Cholesterol       Albumin        Copper      Alk_Phos 
    ##             0           134             0           108           106 
    ##          SGOT Tryglicerides     Platelets   Prothrombin         Stage 
    ##           106           136            11             2             6

``` r
count_unique_df(df[ , cat_vars])
```

    ## [1] "Status 3 of 418"
    ## [1] "Drug 3 of 418"
    ## [1] "Sex 2 of 418"
    ## [1] "Ascites 3 of 418"
    ## [1] "Hepatomegaly 3 of 418"
    ## [1] "Spiders 3 of 418"
    ## [1] "Edema 3 of 418"

``` r
for( n in 1:length(cat_vars)){
  print(cat_vars[n])
  print(table(df[ , cat_vars[n]]))
  }
```

    ## [1] "Status"
    ## 
    ##   C  CL   D 
    ## 232  25 161 
    ## [1] "Drug"
    ## 
    ## D-penicillamine         Placebo 
    ##             158             154 
    ## [1] "Sex"
    ## 
    ##   F   M 
    ## 374  44 
    ## [1] "Ascites"
    ## 
    ##   N   Y 
    ## 288  24 
    ## [1] "Hepatomegaly"
    ## 
    ##   N   Y 
    ## 152 160 
    ## [1] "Spiders"
    ## 
    ##   N   Y 
    ## 222  90 
    ## [1] "Edema"
    ## 
    ##   N   S   Y 
    ## 354  44  20

Criação de uma cópia do data.frame e remoção dos valores NA de variáveis
categóricas:

``` r
df2 <- df[!is.na(df$Ascites), ]
colSums(is.na(df2))
```

    ##            ID        N_Days        Status          Drug           Age 
    ##             0             0             0             0             0 
    ##           Sex       Ascites  Hepatomegaly       Spiders         Edema 
    ##             0             0             0             0             0 
    ##     Bilirubin   Cholesterol       Albumin        Copper      Alk_Phos 
    ##             0            28             0             2             0 
    ##          SGOT Tryglicerides     Platelets   Prothrombin         Stage 
    ##             0            30             4             0             0

A variável “Status” será removida, pois os status “Censored”, “Censored
Due Tx Liver” e “Death” apresentam poucos resultados claros e não
parecem ter uma confiabilidade de informação.

``` r
df2 <- df2[ , !colnames(df2) %in% "Status"]
```

A variável “Edema” aparenta ter uma ordem de importância e será tratada
com ‘label encoding’: No edema, no therapy (N) \< edema without terapy
or solved by it (Y) \< edema despite therapy (S)

``` r
df2$Edema <- ifelse(df2$Edema == "N", 1, 
                    ifelse(df2$Edema == "S", 2, 3))
```

As demais variáveis categóricas possuem apenas dois valores únicos e
serão transformadas em valores binários:

``` r
for (n in 1:length(cat_vars)){
  cat_vars[n] %>% toupper %>% print
  df[ , cat_vars[n]] %>% unique %>% print
  }
```

    ## [1] "STATUS"
    ## [1] "D"  "C"  "CL"
    ## [1] "DRUG"
    ## [1] "D-penicillamine" "Placebo"         NA               
    ## [1] "SEX"
    ## [1] "F" "M"
    ## [1] "ASCITES"
    ## [1] "Y" "N" NA 
    ## [1] "HEPATOMEGALY"
    ## [1] "Y" "N" NA 
    ## [1] "SPIDERS"
    ## [1] "Y" "N" NA 
    ## [1] "EDEMA"
    ## [1] "Y" "N" "S"

``` r
df2$Drug <- ifelse(df2$Drug == "Placebo", 0, 1)

df2$Sex <- ifelse(df2$Sex == "M", 0, 1)
colnames(df2)[colnames(df2) %in% "Sex"] <- "Female"

df2$Ascites <- ifelse(df2$Ascites == "N", 0, 1)

df2$Hepatomegaly <- ifelse(df2$Hepatomegaly == "N", 0, 1)

df2$Spiders <- ifelse(df2$Spiders == "N", 0, 1)

rm(list = ls()[!ls() %in% c("data.dict", "df", "df2", "df_to_factor", 
                            "fill_na_median", "num_vars", "scale_data_frame")])
```

As variáveis numéricas possuem valores NA que serão alterados para o
valor da mediana de cada variável. As variáveis também serão
normalizadas para serem usadas em modelos que possuem essa exigência:

``` r
summary(df2[ , num_vars])
```

    ##        ID             N_Days          Age          Bilirubin     
    ##  Min.   :  1.00   Min.   :  41   Min.   : 9598   Min.   : 0.300  
    ##  1st Qu.: 78.75   1st Qu.:1191   1st Qu.:15428   1st Qu.: 0.800  
    ##  Median :156.50   Median :1840   Median :18188   Median : 1.350  
    ##  Mean   :156.50   Mean   :2006   Mean   :18269   Mean   : 3.256  
    ##  3rd Qu.:234.25   3rd Qu.:2697   3rd Qu.:20715   3rd Qu.: 3.425  
    ##  Max.   :312.00   Max.   :4556   Max.   :28650   Max.   :28.000  
    ##                                                                  
    ##   Cholesterol        Albumin         Copper          Alk_Phos      
    ##  Min.   : 120.0   Min.   :1.96   Min.   :  4.00   Min.   :  289.0  
    ##  1st Qu.: 249.5   1st Qu.:3.31   1st Qu.: 41.25   1st Qu.:  871.5  
    ##  Median : 309.5   Median :3.55   Median : 73.00   Median : 1259.0  
    ##  Mean   : 369.5   Mean   :3.52   Mean   : 97.65   Mean   : 1982.7  
    ##  3rd Qu.: 400.0   3rd Qu.:3.80   3rd Qu.:123.00   3rd Qu.: 1980.0  
    ##  Max.   :1775.0   Max.   :4.64   Max.   :588.00   Max.   :13862.4  
    ##  NA's   :28                      NA's   :2                         
    ##       SGOT        Tryglicerides      Platelets      Prothrombin   
    ##  Min.   : 26.35   Min.   : 33.00   Min.   : 62.0   Min.   : 9.00  
    ##  1st Qu.: 80.60   1st Qu.: 84.25   1st Qu.:199.8   1st Qu.:10.00  
    ##  Median :114.70   Median :108.00   Median :257.0   Median :10.60  
    ##  Mean   :122.56   Mean   :124.70   Mean   :261.9   Mean   :10.73  
    ##  3rd Qu.:151.90   3rd Qu.:151.00   3rd Qu.:322.5   3rd Qu.:11.10  
    ##  Max.   :457.25   Max.   :598.00   Max.   :563.0   Max.   :17.10  
    ##                   NA's   :30       NA's   :4                      
    ##      Stage      
    ##  Min.   :1.000  
    ##  1st Qu.:2.000  
    ##  Median :3.000  
    ##  Mean   :3.032  
    ##  3rd Qu.:4.000  
    ##  Max.   :4.000  
    ## 

``` r
df2[, num_vars] <- sapply(df2[, num_vars], fill_na_median)

df2 <- scale_data_frame(df2, num_vars[!num_vars %in% "Stage"])
```

A variável alvo é a que detalha o estágio dos danos no fígado, sendo o
estágio 4 a cirrose e o estágio 3 o início da fibrose. Os níveis 1 e 2
serão considerados como 0 e os níveis 3 e 4 serão considerados 1:

``` r
df2$Stage <- ifelse(df2$Stage > 2, 0, 1)
```

Transformando as variáveis categóricas para o tipo fator:

``` r
df2 <- df_to_factor(df = df2, 
                    col_list = append(colnames(df2)[!colnames(df2) %in% num_vars], "Stage"))
```

Balanceamento de classes na variável alvo e separação dos dados
restantes em outro data.frame:

``` r
set.seed(123)
table(df2$Stage)
```

    ## 
    ##   0   1 
    ## 229  83

``` r
df2_sample <- rbind(df2[df2$Stage == 1, ]
                    [sample(1:nrow(df2[df2$Stage == 1, ]), size = 83), ], 
                    df2[df2$Stage == 0, ])
```

Dividindo os dados em treino e teste:

``` r
set.seed(123)
idx <- sample(1:nrow(df2_sample), size = 0.8 * nrow(df2_sample))
col_names <- colnames(df2_sample)[!colnames(df2_sample) %in% "Stage"]

x_train <- df2_sample[idx, col_names]
x_test <- df2_sample[-idx, col_names]
y_train <- df2_sample[idx, !colnames(df2_sample) %in% col_names]
y_test <- df2_sample[-idx, !colnames(df2_sample) %in% col_names]
```

Criando e treinando os modelos:

``` r
fitControl= trainControl(method="repeatedcv", 
                         number = 5, 
                         repeats = 5, 
                         savePredictions = "final", 
                         allowParallel = TRUE)

model_list <- caretList(x = x_train, y = y_train, trControl = fitControl, 
                        methodList = c("glm", "adaboost", "cforest", 
                                       "treebag", "ada"), 
                        tuneList = NULL, continue_on_fail = FALSE)
```

    ## Warning in trControlCheck(x = trControl, y = target): indexes not defined in
    ## trControl. Attempting to set them ourselves, so each model in the ensemble will
    ## have the same resampling indexes.

Fazendo as predições:

``` r
pred_glm <- predict.train(model_list$glm, newdata = x_test)
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
```

Comparação entre os modelos:

``` r
summary(result_models)$statistics$Accuracy
```

    ##              Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## GLM     0.5510204 0.6938776 0.7200000 0.7268184 0.7755102 0.8163265    0
    ## ADAB    0.6530612 0.6862745 0.7254902 0.7221628 0.7551020 0.8039216    0
    ## CFOR    0.6400000 0.7346939 0.7551020 0.7454992 0.7755102 0.7959184    0
    ## TREEBAG 0.6200000 0.6800000 0.7058824 0.7132517 0.7600000 0.8039216    0
    ## ADA     0.6800000 0.7142857 0.7400000 0.7374659 0.7551020 0.8163265    0

O modelo com maior acurácia média foi o Conditional Inference Random
Forest:

``` r
fitControl <- trainControl(method = "cv", 
                           number = 10, 
                           savePredictions = "final", 
                           allowParallel = TRUE)

modelo.final <- train(x = df2[ , !colnames(df2) %in% "Stage"], 
                      y = df2[ , "Stage"],
                      trControl = fitControl, 
                      method = "cforest")

modelo.final
```

    ## Conditional Inference Random Forest 
    ## 
    ## 312 samples
    ##  18 predictor
    ##   2 classes: '0', '1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 281, 281, 281, 281, 280, 281, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa      
    ##    2    0.7308468  0.004707717
    ##   10    0.7211694  0.162818036
    ##   18    0.7052419  0.129781621
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.
