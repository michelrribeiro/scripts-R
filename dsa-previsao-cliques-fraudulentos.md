Projeto DSA: Previsão de Cliques Fraudulentos
================
Michel Ribeiro

### Projeto: desenvolver modelo de machine learning para prever se um clique em um anúncio resultará em um download.

#### O dataset utilizado está disponível em: [dataset](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data)

# 

##### Definindo a pasta de trabalho:

``` r
setwd('~/Desktop/DSA/big-Data-R-Azure/projeto_final_01')
```

# 

##### Carregando os pacotes:

``` r
library(dplyr)
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
library(ggplot2)
library(data.table)
```

    ## 
    ## Attaching package: 'data.table'

    ## The following objects are masked from 'package:dplyr':
    ## 
    ##     between, first, last

``` r
library(caret)
```

    ## Loading required package: lattice

# 

##### O dicionário de dados foi copiado e salvo como um arquivo .json para ficar disponível na sessão:

``` r
library(jsonlite)
data_dict <- jsonlite::read_json('./data_fields.json')
detach(package: jsonlite)
data_dict
```

    ## $ip
    ## [1] "ip adress of click."
    ## 
    ## $app
    ## [1] "app id for marketing."
    ## 
    ## $device
    ## [1] "device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc)."
    ## 
    ## $os
    ## [1] "os version id of user mobile phone."
    ## 
    ## $channel
    ## [1] "channel id of mobile ad publisher."
    ## 
    ## $click_time
    ## [1] "timestamp of click (UTC)."
    ## 
    ## $attributed_time
    ## [1] "if user download the app for after clicking an ad, this is the time of the app download."
    ## 
    ## $is_attributed
    ## [1] "the target that is to be predicted, indicating the app was downloaded."
    ## 
    ## $click_id
    ## [1] "reference for making predictions."

# 

##### Como o dataset de dados de treino possui muitos registros, o trabalho inicial será feito em cima da amostra randomizada que já é disponibilizada pelo site:

``` r
dt <- fread('./data/train_sample.csv')
dim(dt)
```

    ## [1] 100000      8

``` r
str(dt)
```

    ## Classes 'data.table' and 'data.frame':   100000 obs. of  8 variables:
    ##  $ ip             : int  87540 105560 101424 94584 68413 93663 17059 121505 192967 143636 ...
    ##  $ app            : int  12 25 12 13 12 3 1 9 2 3 ...
    ##  $ device         : int  1 1 1 1 1 1 1 1 2 1 ...
    ##  $ os             : int  13 17 19 13 1 17 17 25 22 19 ...
    ##  $ channel        : int  497 259 212 477 178 115 135 442 364 135 ...
    ##  $ click_time     : POSIXct, format: "2017-11-07 09:30:38" "2017-11-07 13:40:27" ...
    ##  $ attributed_time: POSIXct, format: NA NA ...
    ##  $ is_attributed  : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  - attr(*, ".internal.selfref")=<externalptr>

``` r
head(dt)
```

    ##        ip app device os channel          click_time attributed_time
    ## 1:  87540  12      1 13     497 2017-11-07 09:30:38            <NA>
    ## 2: 105560  25      1 17     259 2017-11-07 13:40:27            <NA>
    ## 3: 101424  12      1 19     212 2017-11-07 18:05:24            <NA>
    ## 4:  94584  13      1 13     477 2017-11-07 04:58:08            <NA>
    ## 5:  68413  12      1  1     178 2017-11-09 09:00:09            <NA>
    ## 6:  93663   3      1 17     115 2017-11-09 01:22:13            <NA>
    ##    is_attributed
    ## 1:             0
    ## 2:             0
    ## 3:             0
    ## 4:             0
    ## 5:             0
    ## 6:             0

# 

##### Calculando a quantidade de valores NA e valores únicos:

``` r
colSums(is.na(dt))
```

    ##              ip             app          device              os         channel 
    ##               0               0               0               0               0 
    ##      click_time attributed_time   is_attributed 
    ##               0           99773               0

``` r
for (n in c(1:8)){
  colnames(dt)[n] %>% paste(unique(dt[ , ..n]) %>% count) %>% print
  }
```

    ## [1] "ip 34857"
    ## [1] "app 161"
    ## [1] "device 100"
    ## [1] "os 130"
    ## [1] "channel 161"
    ## [1] "click_time 80350"
    ## [1] "attributed_time 228"
    ## [1] "is_attributed 2"

# 

##### Como o horário específico de download possui muitos valores ausentes, a coluna será descartada.

##### A coluna click_time será transformada em uma variável categórica ordinária: 0, 1, 2, 3 para cada intervalo de 6h a partir da meia-noite.

##### Também será removida a variável “ip”, pois não será usada na criação do modelo.

``` r
factor_click_time <- function(number){
  if (number <= 6){0
    } else if (number <= 12){1
    } else if (number <= 18){2
    } else {3}
  }

dt$click_time_fac <- as.integer(format(as.POSIXct(dt$click_time), format = "%H"))
dt$click_time_fac <- sapply(dt$click_time_fac, factor_click_time)

dt <- dt[ , !c("ip", "click_time","attributed_time")]
```

# 

##### Alterando as variáveis para o tipo fator:

``` r
fac_cols <- c(colnames(dt))
dt <- dt[ , (fac_cols) := lapply(.SD, as.factor), .SDcols = fac_cols]

str(dt)
```

    ## Classes 'data.table' and 'data.frame':   100000 obs. of  6 variables:
    ##  $ app           : Factor w/ 161 levels "1","2","3","4",..: 12 25 12 13 12 3 1 9 2 3 ...
    ##  $ device        : Factor w/ 100 levels "0","1","2","4",..: 2 2 2 2 2 2 2 2 3 2 ...
    ##  $ os            : Factor w/ 130 levels "0","1","2","3",..: 14 18 20 14 2 18 18 26 23 20 ...
    ##  $ channel       : Factor w/ 161 levels "3","4","5","13",..: 160 68 53 147 46 21 35 127 101 35 ...
    ##  $ is_attributed : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ click_time_fac: Factor w/ 4 levels "0","1","2","3": 2 3 3 1 2 1 1 2 2 2 ...
    ##  - attr(*, ".internal.selfref")=<externalptr>

``` r
summary(dt)
```

    ##       app            device            os           channel      is_attributed
    ##  3      :18279   1      :94338   19     :23870   280    : 8114   0:99773      
    ##  12     :13198   2      : 4345   13     :21223   245    : 4802   1:  227      
    ##  2      :11737   0      :  541   17     : 5232   107    : 4543                
    ##  9      : 8992   3032   :  371   18     : 4830   477    : 3960                
    ##  15     : 8595   3543   :  151   22     : 4039   134    : 3224                
    ##  18     : 8315   3866   :   93   10     : 2816   259    : 3130                
    ##  (Other):30884   (Other):  161   (Other):37990   (Other):72227                
    ##  click_time_fac
    ##  0:37676       
    ##  1:30711       
    ##  2:23308       
    ##  3: 8305       
    ##                
    ##                
    ## 

# 

##### Separação em amostra de treino e amostra de teste:

``` r
set.seed(16)
idx <- sample(c(rep(0, 0.7 * nrow(dt)), rep(1, 0.3 * nrow(dt))))

train.dt <- dt[which(idx == 0), ]
test.dt <- dt[which(idx == 1), ]

summary(train.dt$is_attributed)
```

    ##     0     1 
    ## 69831   169

# 

##### Balanceando o dataset de treino e diminuindo o volume de dados para conseguir processar e, em seguida, removendo variáveis para liberar memória:

``` r
set.seed(1604)
train.dt2 <- rbind(subset(train.dt, is_attributed == 1), 
             sample_n(subset(train.dt, is_attributed == 0), size = 169))

summary(train.dt2$is_attributed)
```

    ##   0   1 
    ## 169 169

``` r
rm(dt, fac_cols, factor_click_time, idx, n, train.dt)
```

# 

##### Criando o modelo “Boosted Classification Trees”:

``` r
library(ada)
```

    ## Loading required package: rpart

``` r
ada.model <- train(x = train.dt2[, c(1:4, 6)], y = train.dt2$is_attributed, 
                   method = 'ada')
```

# 

##### Aplicando o ada.model aos dados de teste:

``` r
ada.predict <- predict(ada.model, test.dt, type = "raw")

confusionMatrix(table(data = ada.predict, reference = test.dt$is_attributed), positive = '1')
```

    ## Confusion Matrix and Statistics
    ## 
    ##     reference
    ## data     0     1
    ##    0 28368     6
    ##    1  1574    52
    ##                                           
    ##                Accuracy : 0.9473          
    ##                  95% CI : (0.9447, 0.9498)
    ##     No Information Rate : 0.9981          
    ##     P-Value [Acc > NIR] : 1               
    ##                                           
    ##                   Kappa : 0.0582          
    ##                                           
    ##  Mcnemar's Test P-Value : <2e-16          
    ##                                           
    ##             Sensitivity : 0.896552        
    ##             Specificity : 0.947432        
    ##          Pos Pred Value : 0.031980        
    ##          Neg Pred Value : 0.999789        
    ##              Prevalence : 0.001933        
    ##          Detection Rate : 0.001733        
    ##    Detection Prevalence : 0.054200        
    ##       Balanced Accuracy : 0.921992        
    ##                                           
    ##        'Positive' Class : 1               
    ## 

# 

##### A acurácia final encontrada foi acima de 90%, o que foi considerado satisfatório para o primeiro projeto.
