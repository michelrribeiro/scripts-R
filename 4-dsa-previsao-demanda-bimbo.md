Projeto DSA: Previsão de Demanda
================
Michel Ribeiro

### Projeto: desenvolver modelo de machine learning para prever demanda de diversos produtos nas lojas do grupo bimbo.

#### O dataset utilizado está disponível em: [dataset](https://www.kaggle.com/competitions/grupo-bimbo-inventory-demand/data)

Definindo a pasta de trabalho:

``` r
options(warn = -1)
path = "~/Desktop/DSA/big-Data-R-Azure/projeto_final_02"
setwd(paste0(path, "/scripts"))
```

Carregando os pacotes:

``` r
source("~/Desktop/DSA/big-Data-R-Azure/0-functions/functions.R")
library(data.table, quietly = T)
library(dplyr, quietly = T)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:data.table':
    ## 
    ##     between, first, last

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

O dicionário de dados foi copiado e salvo como um arquivo .csv para
ficar disponível na sessão:

``` r
data_dict <- read.csv(paste0(path, "/data.v2/data_fields.csv"), header = T)
View(data_dict)
```

O arquivo a ser trabalhado será uma tabela geral com todos os dados
disponíveis:

``` r
df <- read.csv2(paste0(path, "/data.v2/train.csv"))
townState <- read.csv(paste0(path, "/data.v2/town.state.csv"))
products <- read.csv2(paste0(path, "/data.v2/products.csv"))
customers <- read.csv(paste0(path, "/data.v2/customers.csv"))

colnames(df); colnames(townState); colnames(products); colnames(customers)
```

    ##  [1] "Semana"            "Agencia_ID"        "Canal_ID"         
    ##  [4] "Ruta_SAK"          "Cliente_ID"        "Producto_ID"      
    ##  [7] "Venta_uni_hoy"     "Venta_hoy"         "Dev_uni_proxima"  
    ## [10] "Dev_proxima"       "Demanda_uni_equil"

    ## [1] "Agencia_ID" "Town"       "State"

    ## [1] "Producto_ID"    "NombreProducto"

    ## [1] "Cliente_ID"    "NombreCliente"

``` r
df2 <- df %>% inner_join(townState, suffix = c("Agencia_ID", "Agencia_ID")) %>% 
  inner_join(products, suffix = c("Producto_ID", "Producto_ID")) %>% 
  inner_join(customers, suffix = c("Cliente_ID", "CLiente_ID"))
```

    ## Joining, by = "Agencia_ID"
    ## Joining, by = "Producto_ID"
    ## Joining, by = "Cliente_ID"

``` r
colSums(is.na(df2))
```

    ##            Semana        Agencia_ID          Canal_ID          Ruta_SAK 
    ##                 0                 0                 0                 0 
    ##        Cliente_ID       Producto_ID     Venta_uni_hoy         Venta_hoy 
    ##                 0                 0                 0                 0 
    ##   Dev_uni_proxima       Dev_proxima Demanda_uni_equil              Town 
    ##                 0                 0                 0                 0 
    ##             State    NombreProducto     NombreCliente 
    ##                 0                 0                 0

``` r
df2[(df2$Semana == 0 | df2$Agencia_ID == 0 | df2$Canal_ID == 0 | df2$Ruta_SAK == 0 | 
      df2$Producto_ID == 0), ]
```

    ##  [1] Semana            Agencia_ID        Canal_ID          Ruta_SAK         
    ##  [5] Cliente_ID        Producto_ID       Venta_uni_hoy     Venta_hoy        
    ##  [9] Dev_uni_proxima   Dev_proxima       Demanda_uni_equil Town             
    ## [13] State             NombreProducto    NombreCliente    
    ## <0 rows> (or 0-length row.names)

Remoção das colunas ID, State e NombreCliente, pois a cidade já carrega
a informação de região e o nome do cliente seria mais útil em uma
avaliação visando conhecer os padrões de compra específicos:

``` r
rm_col <- c("Agencia_ID", "Cliente_ID", "Producto_ID", "NombreCliente", 
            "Ruta_SAK", "State")
df2 <- df2[ , !colnames(df2) %in% rm_col]
```

Como os dados de peso por item serão mantidos no nome do item, as
colunas com a venda em unidades de peso e o que sobrou para a próxima
semana em peso serão removidas:

``` r
df2 <- df2[ , !colnames(df2) %in% c("Venta_hoy", "Dev_proxima")]
```

Tratamento das colunas restantes:

``` r
glimpse(df2)
```

    ## Rows: 20,000
    ## Columns: 7
    ## $ Semana            <int> 6, 7, 6, 9, 6, 5, 4, 7, 6, 4, 5, 5, 7, 5, 4, 8, 9, 5…
    ## $ Canal_ID          <int> 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 11, 1, 1, 1, 1, …
    ## $ Venta_uni_hoy     <int> 2, 2, 8, 2, 10, 1, 5, 3, 1, 3, 1, 3, 5, 12, 4, 3, 2,…
    ## $ Dev_uni_proxima   <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0…
    ## $ Demanda_uni_equil <int> 2, 2, 8, 2, 10, 1, 5, 3, 1, 3, 1, 3, 5, 12, 4, 3, 2,…
    ## $ Town              <chr> "CUERNAVACA CIVAC", "JALAPA I", "PUEBLA SUR BIMBO", …
    ## $ NombreProducto    <chr> "DONAS AZUCAR 4P 105G BIM", "GANSITO 1P 50G CCHAROLA…

``` r
rm_col2 <- c("Venta_uni_hoy", "Venta_hoy", "Dev_uni_proxima", "Dev_proxima", "Demanda_uni_equil")
df2[ , !colnames(df2) %in% rm_col2] %>% as.data.table %>% count_unique_dt
```

    ## [1] "Semana 1 of 20000"
    ## [1] "Canal_ID 1 of 20000"
    ## [1] "Town 1 of 20000"
    ## [1] "NombreProducto 1 of 20000"

A demanda da próxima semana é dada por (itens vendidos - itens
restantes), sendo que a demanda da próxima semanana não pode ser
negativa.  
Aplicando One Hot Encoding às variáveis “Town” e “NombreProducto”:

``` r
library(CatEncoders)
```

    ## 
    ## Attaching package: 'CatEncoders'

    ## The following object is masked from 'package:base':
    ## 
    ##     transform

``` r
df_Town <- slot(LabelEncoder.fit(df2$Town), name = "mapping")
colnames(df_Town) <- c("Town", "ind_Town")

df_NombreProducto <- slot(LabelEncoder.fit(df2$NombreProducto), name = "mapping")
colnames(df_NombreProducto) <- c("NombreProducto", "ind_Prod")

df2 <- df2 %>% inner_join(df_Town, suffix = c("Town", "Town")) %>% 
  inner_join(df_NombreProducto, suffix = c("NombreProducto", "NombreProducto"))
```

    ## Joining, by = "Town"

    ## Joining, by = "NombreProducto"

``` r
df2 <- df2[ , !colnames(df2) %in% c("Town", "NombreProducto")]
```

Como a previsão será feita baseada nas cidades e nos produtos, a
primeira tentativa será feita sem a variável “Canal_ID”:

``` r
set.seed(1234)
idx <- sample(1:nrow(df2), size = 1000)
df4 <- df2[idx, ]
```

Dividindo os dados:

``` r
set.seed(1234)
idx <- sample(1:nrow(df4), size = 0.8 * nrow(df4))
x_train <- df4[idx, !colnames(df4) %in% c("Demanda_uni_equil")]
y_train <- df4[idx, c("Demanda_uni_equil")]
x_test <- df4[-idx, !colnames(df4) %in% c("Demanda_uni_equil")]
y_test <- df4[-idx, c("Demanda_uni_equil")]

rm(list = ls()[!ls() %in% c("df2", "path", "scale_data_frame", "x_train", 
                            "x_test", "y_train", "y_test")])
```

Treinando os modelos escolhidos com caretEnsemble:

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
fitControl= trainControl(method="cv", number = 5, 
                         savePredictions = "final", allowParallel = TRUE)

model_list <- caretList(x = x_train, y = y_train, trControl = fitControl, 
                        methodList = c("lm", "svmRadial", "rf", "bridge", "xgbLinear"),
                        tuneList = NULL, continue_on_fail = FALSE)
```

    ## t=100, m=2
    ## t=200, m=2
    ## t=300, m=2
    ## t=400, m=2
    ## t=500, m=2
    ## t=600, m=2
    ## t=700, m=2
    ## t=800, m=2
    ## t=900, m=2
    ## t=100, m=2
    ## t=200, m=2
    ## t=300, m=2
    ## t=400, m=2
    ## t=500, m=2
    ## t=600, m=2
    ## t=700, m=2
    ## t=800, m=2
    ## t=900, m=2
    ## t=100, m=2
    ## t=200, m=2
    ## t=300, m=2
    ## t=400, m=2
    ## t=500, m=2
    ## t=600, m=2
    ## t=700, m=2
    ## t=800, m=2
    ## t=900, m=2
    ## t=100, m=2
    ## t=200, m=2
    ## t=300, m=2
    ## t=400, m=2
    ## t=500, m=2
    ## t=600, m=2
    ## t=700, m=2
    ## t=800, m=2
    ## t=900, m=2
    ## t=100, m=2
    ## t=200, m=2
    ## t=300, m=2
    ## t=400, m=2
    ## t=500, m=2
    ## t=600, m=2
    ## t=700, m=2
    ## t=800, m=2
    ## t=900, m=2
    ## t=100, m=2
    ## t=200, m=2
    ## t=300, m=2
    ## t=400, m=2
    ## t=500, m=2
    ## t=600, m=2
    ## t=700, m=2
    ## t=800, m=2
    ## t=900, m=2

Fazendo as predições:

``` r
pred_lm <- predict.train(model_list$lm, newdata = x_test) %>% round()
pred_svm <- predict.train(model_list$svmRadial, newdata = x_test) %>% round()
pred_rf <- predict.train(model_list$rf, newdata = x_test) %>% round()
pred_bridge <- predict.train(model_list$bridge, newdata = x_test) %>% round()
pred_xgbL <- predict.train(model_list$xgbLinear, newdata = x_test) %>% round()
```

Avaliando os modelos:

``` r
result_models <- 
  data.frame(LM = c(min(model_list$lm$results$RMSE), RMSE(pred_lm, y_test), cor(pred_lm, y_test)), 
             SVM = c(min(model_list$svmRadial$results$RMSE), RMSE(pred_svm, y_test), cor(pred_svm, y_test)), 
             RF = c(min(model_list$rf$results$RMSE), RMSE(pred_rf, y_test), cor(pred_rf, y_test)), 
             BRIDGE = c(min(model_list$bridge$results$RMSE), RMSE(pred_bridge, y_test), cor(pred_bridge, y_test)), 
             XGBL = c(min(model_list$xgbLinear$results$RMSE), RMSE(pred_xgbL, y_test), cor(pred_xgbL, y_test)))

rownames(result_models) <- c("RMSE_model", "RMSE_predict", "Corr_Pred_Real")

result_models
```

    ##                       LM        SVM        RF    BRIDGE      XGBL
    ## RMSE_model     0.1439417 16.7670441 5.6460087 0.1701182 5.0567200
    ## RMSE_predict   0.1224745 12.9813328 3.4713110 0.1732051 1.4747881
    ## Corr_Pred_Real 0.9999735  0.6802672 0.9918647 0.9999470 0.9980235

``` r
rm(list = ls()[!ls() %in% c("df2", "fitControl", "result_models")])
```

Treinando o modelo ‘lm’ com um maior volume de dados:

``` r
model_list <- caretList(x = df2[ , !colnames(df2) %in% c("Demanda_uni_equil")], 
                        y = df2[ , c("Demanda_uni_equil")], trControl = fitControl, 
                        methodList = c("lm"), tuneList = NULL, continue_on_fail = FALSE)

modelo_final <- model_list$lm
```
