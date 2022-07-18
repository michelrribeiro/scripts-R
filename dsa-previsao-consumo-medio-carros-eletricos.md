Projeto DSA: Previsão do Consumo Médio de Carros Elétricos
================
Michel Ribeiro

### Projeto: desenvolver modelo de machine learning para prever qual a eficiência de um carro elétrico com base em suas características.

#### O dataset utilizado está disponível em: [dataset](https://data.mendeley.com/datasets/tb9yrptydn/2)

# 

##### Definindo a pasta de trabalho:

``` r
setwd("~/Desktop/DSA/big-Data-R-Azure/projeto_final_01.v02")
```

# 

##### Carregando os pacotes:

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
source("~/Desktop/DSA/big-Data-R-Azure/0-functions/functions.R")
```

# 

##### Carregando o dataset:

``` r
library(readxl, quietly = T)
df <- readxl::read_xlsx('./FEV-data-Excel.xlsx') %>% as.data.frame
detach(package: readxl)
```

# 

##### Visualizando subset com linhas que possuem valores em branco:

``` r
glimpse(df)
```

    ## Rows: 53
    ## Columns: 25
    ## $ `Car full name`                          <chr> "Audi e-tron 55 quattro", "Au…
    ## $ Make                                     <chr> "Audi", "Audi", "Audi", "Audi…
    ## $ Model                                    <chr> "e-tron 55 quattro", "e-tron …
    ## $ `Minimal price (gross) [PLN]`            <dbl> 345700, 308400, 414900, 31970…
    ## $ `Engine power [KM]`                      <dbl> 360, 313, 503, 313, 360, 503,…
    ## $ `Maximum torque [Nm]`                    <dbl> 664, 540, 973, 540, 664, 973,…
    ## $ `Type of brakes`                         <chr> "disc (front + rear)", "disc …
    ## $ `Drive type`                             <chr> "4WD", "4WD", "4WD", "4WD", "…
    ## $ `Battery capacity [kWh]`                 <dbl> 95.0, 71.0, 95.0, 71.0, 95.0,…
    ## $ `Range (WLTP) [km]`                      <dbl> 438, 340, 364, 346, 447, 369,…
    ## $ `Wheelbase [cm]`                         <dbl> 292.8, 292.8, 292.8, 292.8, 2…
    ## $ `Length [cm]`                            <dbl> 490.1, 490.1, 490.2, 490.1, 4…
    ## $ `Width [cm]`                             <dbl> 193.5, 193.5, 197.6, 193.5, 1…
    ## $ `Height [cm]`                            <dbl> 162.9, 162.9, 162.9, 161.6, 1…
    ## $ `Minimal empty weight [kg]`              <dbl> 2565, 2445, 2695, 2445, 2595,…
    ## $ `Permissable gross weight [kg]`          <dbl> 3130, 3040, 3130, 3040, 3130,…
    ## $ `Maximum load capacity [kg]`             <dbl> 640, 670, 565, 640, 670, 565,…
    ## $ `Number of seats`                        <dbl> 5, 5, 5, 5, 5, 5, 4, 4, 5, 5,…
    ## $ `Number of doors`                        <dbl> 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,…
    ## $ `Tire size [in]`                         <dbl> 19, 19, 20, 19, 19, 20, 19, 2…
    ## $ `Maximum speed [kph]`                    <dbl> 200, 190, 210, 190, 200, 210,…
    ## $ `Boot capacity (VDA) [l]`                <dbl> 660, 660, 660, 615, 615, 615,…
    ## $ `Acceleration 0-100 kph [s]`             <dbl> 5.7, 6.8, 4.5, 6.8, 5.7, 4.5,…
    ## $ `Maximum DC charging power [kW]`         <dbl> 150, 150, 150, 150, 150, 150,…
    ## $ `mean - Energy consumption [kWh/100 km]` <dbl> 24.45, 23.80, 27.55, 23.30, 2…

``` r
colSums(is.na(df))[(colSums(is.na(df)) > 0)]
```

    ##                         Type of brakes          Permissable gross weight [kg] 
    ##                                      1                                      8 
    ##             Maximum load capacity [kg]                Boot capacity (VDA) [l] 
    ##                                      8                                      1 
    ##             Acceleration 0-100 kph [s] mean - Energy consumption [kWh/100 km] 
    ##                                      3                                      9

``` r
df[rowSums(is.na(df)) > 0, ]
```

    ##                        Car full name          Make                       Model
    ## 10                      Citroën ë-C4       Citroën                        ë-C4
    ## 30                    Peugeot e-2008       Peugeot                      e-2008
    ## 40 Tesla Model 3 Standard Range Plus         Tesla Model 3 Standard Range Plus
    ## 41          Tesla Model 3 Long Range         Tesla          Model 3 Long Range
    ## 42         Tesla Model 3 Performance         Tesla         Model 3 Performance
    ## 43     Tesla Model S Long Range Plus         Tesla     Model S Long Range Plus
    ## 44         Tesla Model S Performance         Tesla         Model S Performance
    ## 45     Tesla Model X Long Range Plus         Tesla     Model X Long Range Plus
    ## 46         Tesla Model X Performance         Tesla         Model X Performance
    ## 52          Mercedes-Benz EQV (long) Mercedes-Benz                  EQV (long)
    ## 53             Nissan e-NV200 evalia        Nissan              e-NV200 evalia
    ##    Minimal price (gross) [PLN] Engine power [KM] Maximum torque [Nm]
    ## 10                      125000               136                 260
    ## 30                      149400               136                 260
    ## 40                      195490               285                 450
    ## 41                      235490               372                 510
    ## 42                      260490               480                 639
    ## 43                      368990               525                 755
    ## 44                      443990               772                1140
    ## 45                      407990               525                 755
    ## 46                      482990               772                1140
    ## 52                      339480               204                 362
    ## 53                      164328               109                 254
    ##         Type of brakes  Drive type Battery capacity [kWh] Range (WLTP) [km]
    ## 10 disc (front + rear) 2WD (front)                     50               350
    ## 30 disc (front + rear) 2WD (front)                     50               320
    ## 40 disc (front + rear)  2WD (rear)                     54               430
    ## 41 disc (front + rear)         4WD                     75               580
    ## 42 disc (front + rear)         4WD                     75               567
    ## 43 disc (front + rear)         4WD                    100               652
    ## 44 disc (front + rear)         4WD                    100               639
    ## 45 disc (front + rear)         4WD                    100               561
    ## 46 disc (front + rear)         4WD                    100               548
    ## 52                <NA> 2WD (front)                     90               356
    ## 53 disc (front + rear) 2WD (front)                     40               200
    ##    Wheelbase [cm] Length [cm] Width [cm] Height [cm] Minimal empty weight [kg]
    ## 10          266.7       435.4      180.0       152.2                      1541
    ## 30          260.5       430.0      177.0       153.0                      1548
    ## 40          287.5       469.0      193.0       144.0                      1626
    ## 41          287.5       469.0      193.0       144.0                      1862
    ## 42          287.5       469.0      193.0       144.0                      1862
    ## 43          296.0       497.9      196.4       144.5                      2391
    ## 44          296.0       497.9      196.4       144.5                      2417
    ## 45          296.5       503.7      207.0       162.6                      2464
    ## 46          296.5       503.7      207.0       162.6                      2524
    ## 52          320.0       514.0      192.8       191.0                      2710
    ## 53          272.5       456.0      175.5       185.8                      1592
    ##    Permissable gross weight [kg] Maximum load capacity [kg] Number of seats
    ## 10                          2000                        459               5
    ## 30                            NA                         NA               5
    ## 40                            NA                         NA               5
    ## 41                            NA                         NA               5
    ## 42                            NA                         NA               5
    ## 43                            NA                         NA               5
    ## 44                            NA                         NA               5
    ## 45                            NA                         NA               7
    ## 46                            NA                         NA               7
    ## 52                          3500                        865               6
    ## 53                          2250                        658               5
    ##    Number of doors Tire size [in] Maximum speed [kph] Boot capacity (VDA) [l]
    ## 10               5             16                 150                     380
    ## 30               5             16                 150                     434
    ## 40               5             18                 225                     425
    ## 41               5             18                 233                     425
    ## 42               5             20                 261                     425
    ## 43               5             19                 250                     745
    ## 44               5             21                 261                     745
    ## 45               5             20                 250                     857
    ## 46               5             20                 261                     857
    ## 52               5             17                 160                      NA
    ## 53               5             15                 123                     870
    ##    Acceleration 0-100 kph [s] Maximum DC charging power [kW]
    ## 10                        9.5                            100
    ## 30                         NA                            100
    ## 40                        5.6                            150
    ## 41                        4.4                            150
    ## 42                        3.3                            150
    ## 43                        3.8                            150
    ## 44                        2.5                            150
    ## 45                        4.6                            150
    ## 46                        2.8                            150
    ## 52                         NA                            110
    ## 53                         NA                             50
    ##    mean - Energy consumption [kWh/100 km]
    ## 10                                     NA
    ## 30                                     NA
    ## 40                                     NA
    ## 41                                     NA
    ## 42                                     NA
    ## 43                                     NA
    ## 44                                     NA
    ## 45                                     NA
    ## 46                                     NA
    ## 52                                   28.2
    ## 53                                   25.9

# 

##### Criando subset sem os valores em branco:

``` r
df2 <- df[rowSums(is.na(df)) == 0, ]
```

# 

##### Renomeando colunas para remover espaços e evitar problemas nos modelos:

``` r
colnames(df2) <- gsub(" ", "", colnames(df2))
colnames(df2)
```

    ##  [1] "Carfullname"                       "Make"                             
    ##  [3] "Model"                             "Minimalprice(gross)[PLN]"         
    ##  [5] "Enginepower[KM]"                   "Maximumtorque[Nm]"                
    ##  [7] "Typeofbrakes"                      "Drivetype"                        
    ##  [9] "Batterycapacity[kWh]"              "Range(WLTP)[km]"                  
    ## [11] "Wheelbase[cm]"                     "Length[cm]"                       
    ## [13] "Width[cm]"                         "Height[cm]"                       
    ## [15] "Minimalemptyweight[kg]"            "Permissablegrossweight[kg]"       
    ## [17] "Maximumloadcapacity[kg]"           "Numberofseats"                    
    ## [19] "Numberofdoors"                     "Tiresize[in]"                     
    ## [21] "Maximumspeed[kph]"                 "Bootcapacity(VDA)[l]"             
    ## [23] "Acceleration0-100kph[s]"           "MaximumDCchargingpower[kW]"       
    ## [25] "mean-Energyconsumption[kWh/100km]"

# 

##### Criando subset sem valores em branco e com variáveis numéricas apenas:

``` r
col_to_drop <- c("Carfullname", "Make", "Model", "Typeofbrakes", "Drivetype")
df2_num <- df2[ , !colnames(df2) %in% col_to_drop]

dim(df2_num)
```

    ## [1] 42 20

``` r
summary(df2_num)
```

    ##  Minimalprice(gross)[PLN] Enginepower[KM] Maximumtorque[Nm]
    ##  Min.   : 82050           Min.   : 82.0   Min.   : 160.0   
    ##  1st Qu.:140650           1st Qu.:136.0   1st Qu.: 260.0   
    ##  Median :166945           Median :184.0   Median : 317.5   
    ##  Mean   :235066           Mean   :237.7   Mean   : 425.2   
    ##  3rd Qu.:316875           3rd Qu.:313.0   3rd Qu.: 540.0   
    ##  Max.   :794000           Max.   :625.0   Max.   :1050.0   
    ##  Batterycapacity[kWh] Range(WLTP)[km] Wheelbase[cm]     Length[cm]   
    ##  Min.   :17.60        Min.   :148.0   Min.   :187.3   Min.   :269.5  
    ##  1st Qu.:39.20        1st Qu.:279.2   1st Qu.:256.3   1st Qu.:406.6  
    ##  Median :52.00        Median :352.5   Median :270.0   Median :431.8  
    ##  Mean   :58.84        Mean   :351.7   Mean   :269.8   Mean   :433.5  
    ##  3rd Qu.:78.65        3rd Qu.:434.8   3rd Qu.:290.0   3rd Qu.:475.5  
    ##  Max.   :95.00        Max.   :549.0   Max.   :327.5   Max.   :496.3  
    ##    Width[cm]       Height[cm]    Minimalemptyweight[kg]
    ##  Min.   :164.5   Min.   :137.8   Min.   :1035          
    ##  1st Qu.:178.7   1st Qu.:151.2   1st Qu.:1516          
    ##  Median :180.2   Median :156.0   Median :1622          
    ##  Mean   :184.8   Mean   :155.0   Mean   :1821          
    ##  3rd Qu.:193.5   3rd Qu.:160.5   3rd Qu.:2249          
    ##  Max.   :255.8   Max.   :190.0   Max.   :2695          
    ##  Permissablegrossweight[kg] Maximumloadcapacity[kg] Numberofseats  
    ##  Min.   :1310               Min.   : 290.0          Min.   :2.000  
    ##  1st Qu.:1882               1st Qu.: 440.0          1st Qu.:4.250  
    ##  Median :2100               Median : 485.5          Median :5.000  
    ##  Mean   :2268               Mean   : 510.5          Mean   :4.762  
    ##  3rd Qu.:2855               3rd Qu.: 565.0          3rd Qu.:5.000  
    ##  Max.   :3130               Max.   :1056.0          Max.   :8.000  
    ##  Numberofdoors   Tiresize[in]   Maximumspeed[kph] Bootcapacity(VDA)[l]
    ##  Min.   :3.00   Min.   :14.00   Min.   :130.0     Min.   :171.0       
    ##  1st Qu.:5.00   1st Qu.:16.00   1st Qu.:146.2     1st Qu.:310.2       
    ##  Median :5.00   Median :17.00   Median :160.0     Median :371.0       
    ##  Mean   :4.81   Mean   :17.55   Mean   :169.5     Mean   :404.3       
    ##  3rd Qu.:5.00   3rd Qu.:19.00   3rd Qu.:187.5     3rd Qu.:497.0       
    ##  Max.   :5.00   Max.   :21.00   Max.   :260.0     Max.   :660.0       
    ##  Acceleration0-100kph[s] MaximumDCchargingpower[kW]
    ##  Min.   : 2.800          Min.   : 22.0             
    ##  1st Qu.: 6.800          1st Qu.: 62.5             
    ##  Median : 7.900          Median :100.0             
    ##  Mean   : 7.893          Mean   :109.7             
    ##  3rd Qu.: 9.650          3rd Qu.:143.8             
    ##  Max.   :13.100          Max.   :270.0             
    ##  mean-Energyconsumption[kWh/100km]
    ##  Min.   :13.10                    
    ##  1st Qu.:15.60                    
    ##  Median :16.88                    
    ##  Mean   :18.61                    
    ##  3rd Qu.:22.94                    
    ##  Max.   :27.55

# 

##### Avaliando correlação:

``` r
library(reshape2, quietly = T)
corr_matrix <- data.frame(melt(cor(df2_num)), stringsAsFactors = F)
detach(package: reshape2)
```

# 

##### Correlação das variáveis preditoras entre si:

``` r
corr_matrix[corr_matrix$Var2 != "mean-Energyconsumption[kWh/100km]" & 
                   corr_matrix$Var1 != "mean-Energyconsumption[kWh/10km]" &
                   corr_matrix$Var1 != corr_matrix$Var2, ]
```

    ##                                  Var1                       Var2        value
    ## 2                     Enginepower[KM]   Minimalprice(gross)[PLN]  0.960909547
    ## 3                   Maximumtorque[Nm]   Minimalprice(gross)[PLN]  0.902694431
    ## 4                Batterycapacity[kWh]   Minimalprice(gross)[PLN]  0.794080684
    ## 5                     Range(WLTP)[km]   Minimalprice(gross)[PLN]  0.457710170
    ## 6                       Wheelbase[cm]   Minimalprice(gross)[PLN]  0.624353697
    ## 7                          Length[cm]   Minimalprice(gross)[PLN]  0.734752249
    ## 8                           Width[cm]   Minimalprice(gross)[PLN]  0.476217707
    ## 9                          Height[cm]   Minimalprice(gross)[PLN] -0.235877258
    ## 10             Minimalemptyweight[kg]   Minimalprice(gross)[PLN]  0.786707488
    ## 11         Permissablegrossweight[kg]   Minimalprice(gross)[PLN]  0.767760358
    ## 12            Maximumloadcapacity[kg]   Minimalprice(gross)[PLN]  0.475068582
    ## 13                      Numberofseats   Minimalprice(gross)[PLN] -0.090688436
    ## 14                      Numberofdoors   Minimalprice(gross)[PLN] -0.306692899
    ## 15                       Tiresize[in]   Minimalprice(gross)[PLN]  0.750241486
    ## 16                  Maximumspeed[kph]   Minimalprice(gross)[PLN]  0.943811567
    ## 17               Bootcapacity(VDA)[l]   Minimalprice(gross)[PLN]  0.575947547
    ## 18            Acceleration0-100kph[s]   Minimalprice(gross)[PLN] -0.810990436
    ## 19         MaximumDCchargingpower[kW]   Minimalprice(gross)[PLN]  0.887719302
    ## 20  mean-Energyconsumption[kWh/100km]   Minimalprice(gross)[PLN]  0.799447422
    ## 21           Minimalprice(gross)[PLN]            Enginepower[KM]  0.960909547
    ## 23                  Maximumtorque[Nm]            Enginepower[KM]  0.952062140
    ## 24               Batterycapacity[kWh]            Enginepower[KM]  0.868520614
    ## 25                    Range(WLTP)[km]            Enginepower[KM]  0.537007629
    ## 26                      Wheelbase[cm]            Enginepower[KM]  0.644983826
    ## 27                         Length[cm]            Enginepower[KM]  0.773156500
    ## 28                          Width[cm]            Enginepower[KM]  0.471872177
    ## 29                         Height[cm]            Enginepower[KM] -0.193816903
    ## 30             Minimalemptyweight[kg]            Enginepower[KM]  0.855334110
    ## 31         Permissablegrossweight[kg]            Enginepower[KM]  0.815072787
    ## 32            Maximumloadcapacity[kg]            Enginepower[KM]  0.432707985
    ## 33                      Numberofseats            Enginepower[KM] -0.087453650
    ## 34                      Numberofdoors            Enginepower[KM] -0.260890205
    ## 35                       Tiresize[in]            Enginepower[KM]  0.800977748
    ## 36                  Maximumspeed[kph]            Enginepower[KM]  0.951682772
    ## 37               Bootcapacity(VDA)[l]            Enginepower[KM]  0.644273618
    ## 38            Acceleration0-100kph[s]            Enginepower[KM] -0.896693568
    ## 39         MaximumDCchargingpower[kW]            Enginepower[KM]  0.854077056
    ## 40  mean-Energyconsumption[kWh/100km]            Enginepower[KM]  0.824648431
    ## 41           Minimalprice(gross)[PLN]          Maximumtorque[Nm]  0.902694431
    ## 42                    Enginepower[KM]          Maximumtorque[Nm]  0.952062140
    ## 44               Batterycapacity[kWh]          Maximumtorque[Nm]  0.831894406
    ## 45                    Range(WLTP)[km]          Maximumtorque[Nm]  0.450305128
    ## 46                      Wheelbase[cm]          Maximumtorque[Nm]  0.635060942
    ## 47                         Length[cm]          Maximumtorque[Nm]  0.760928943
    ## 48                          Width[cm]          Maximumtorque[Nm]  0.457055008
    ## 49                         Height[cm]          Maximumtorque[Nm] -0.092389876
    ## 50             Minimalemptyweight[kg]          Maximumtorque[Nm]  0.858799330
    ## 51         Permissablegrossweight[kg]          Maximumtorque[Nm]  0.803782763
    ## 52            Maximumloadcapacity[kg]          Maximumtorque[Nm]  0.390599129
    ## 53                      Numberofseats          Maximumtorque[Nm] -0.001363035
    ## 54                      Numberofdoors          Maximumtorque[Nm] -0.133803691
    ## 55                       Tiresize[in]          Maximumtorque[Nm]  0.749802241
    ## 56                  Maximumspeed[kph]          Maximumtorque[Nm]  0.877643876
    ## 57               Bootcapacity(VDA)[l]          Maximumtorque[Nm]  0.676182162
    ## 58            Acceleration0-100kph[s]          Maximumtorque[Nm] -0.832416924
    ## 59         MaximumDCchargingpower[kW]          Maximumtorque[Nm]  0.770809047
    ## 60  mean-Energyconsumption[kWh/100km]          Maximumtorque[Nm]  0.827639009
    ## 61           Minimalprice(gross)[PLN]       Batterycapacity[kWh]  0.794080684
    ## 62                    Enginepower[KM]       Batterycapacity[kWh]  0.868520614
    ## 63                  Maximumtorque[Nm]       Batterycapacity[kWh]  0.831894406
    ## 65                    Range(WLTP)[km]       Batterycapacity[kWh]  0.809255264
    ## 66                      Wheelbase[cm]       Batterycapacity[kWh]  0.744602027
    ## 67                         Length[cm]       Batterycapacity[kWh]  0.841836833
    ## 68                          Width[cm]       Batterycapacity[kWh]  0.524706737
    ## 69                         Height[cm]       Batterycapacity[kWh]  0.041634232
    ## 70             Minimalemptyweight[kg]       Batterycapacity[kWh]  0.920965051
    ## 71         Permissablegrossweight[kg]       Batterycapacity[kWh]  0.885011107
    ## 72            Maximumloadcapacity[kg]       Batterycapacity[kWh]  0.507539014
    ## 73                      Numberofseats       Batterycapacity[kWh]  0.167248289
    ## 74                      Numberofdoors       Batterycapacity[kWh]  0.038005880
    ## 75                       Tiresize[in]       Batterycapacity[kWh]  0.788210908
    ## 76                  Maximumspeed[kph]       Batterycapacity[kWh]  0.823871149
    ## 77               Bootcapacity(VDA)[l]       Batterycapacity[kWh]  0.794238773
    ## 78            Acceleration0-100kph[s]       Batterycapacity[kWh] -0.819402510
    ## 79         MaximumDCchargingpower[kW]       Batterycapacity[kWh]  0.788901752
    ## 80  mean-Energyconsumption[kWh/100km]       Batterycapacity[kWh]  0.757865744
    ## 81           Minimalprice(gross)[PLN]            Range(WLTP)[km]  0.457710170
    ## 82                    Enginepower[KM]            Range(WLTP)[km]  0.537007629
    ## 83                  Maximumtorque[Nm]            Range(WLTP)[km]  0.450305128
    ## 84               Batterycapacity[kWh]            Range(WLTP)[km]  0.809255264
    ## 86                      Wheelbase[cm]            Range(WLTP)[km]  0.507544520
    ## 87                         Length[cm]            Range(WLTP)[km]  0.604403819
    ## 88                          Width[cm]            Range(WLTP)[km]  0.336690121
    ## 89                         Height[cm]            Range(WLTP)[km] -0.011242703
    ## 90             Minimalemptyweight[kg]            Range(WLTP)[km]  0.591383801
    ## 91         Permissablegrossweight[kg]            Range(WLTP)[km]  0.549737187
    ## 92            Maximumloadcapacity[kg]            Range(WLTP)[km]  0.273642010
    ## 93                      Numberofseats            Range(WLTP)[km]  0.130506408
    ## 94                      Numberofdoors            Range(WLTP)[km]  0.152500139
    ## 95                       Tiresize[in]            Range(WLTP)[km]  0.618844536
    ## 96                  Maximumspeed[kph]            Range(WLTP)[km]  0.561107675
    ## 97               Bootcapacity(VDA)[l]            Range(WLTP)[km]  0.508850248
    ## 98            Acceleration0-100kph[s]            Range(WLTP)[km] -0.627631213
    ## 99         MaximumDCchargingpower[kW]            Range(WLTP)[km]  0.551216206
    ## 100 mean-Energyconsumption[kWh/100km]            Range(WLTP)[km]  0.274489077
    ## 101          Minimalprice(gross)[PLN]              Wheelbase[cm]  0.624353697
    ## 102                   Enginepower[KM]              Wheelbase[cm]  0.644983826
    ## 103                 Maximumtorque[Nm]              Wheelbase[cm]  0.635060942
    ## 104              Batterycapacity[kWh]              Wheelbase[cm]  0.744602027
    ## 105                   Range(WLTP)[km]              Wheelbase[cm]  0.507544520
    ## 107                        Length[cm]              Wheelbase[cm]  0.913716463
    ## 108                         Width[cm]              Wheelbase[cm]  0.488064510
    ## 109                        Height[cm]              Wheelbase[cm]  0.328453438
    ## 110            Minimalemptyweight[kg]              Wheelbase[cm]  0.828073460
    ## 111        Permissablegrossweight[kg]              Wheelbase[cm]  0.867939800
    ## 112           Maximumloadcapacity[kg]              Wheelbase[cm]  0.810573337
    ## 113                     Numberofseats              Wheelbase[cm]  0.625667199
    ## 114                     Numberofdoors              Wheelbase[cm]  0.264680729
    ## 115                      Tiresize[in]              Wheelbase[cm]  0.643339062
    ## 116                 Maximumspeed[kph]              Wheelbase[cm]  0.595937406
    ## 117              Bootcapacity(VDA)[l]              Wheelbase[cm]  0.844244547
    ## 118           Acceleration0-100kph[s]              Wheelbase[cm] -0.537228280
    ## 119        MaximumDCchargingpower[kW]              Wheelbase[cm]  0.627443514
    ## 120 mean-Energyconsumption[kWh/100km]              Wheelbase[cm]  0.716478122
    ## 121          Minimalprice(gross)[PLN]                 Length[cm]  0.734752249
    ## 122                   Enginepower[KM]                 Length[cm]  0.773156500
    ## 123                 Maximumtorque[Nm]                 Length[cm]  0.760928943
    ## 124              Batterycapacity[kWh]                 Length[cm]  0.841836833
    ## 125                   Range(WLTP)[km]                 Length[cm]  0.604403819
    ## 126                     Wheelbase[cm]                 Length[cm]  0.913716463
    ## 128                         Width[cm]                 Length[cm]  0.543547107
    ## 129                        Height[cm]                 Length[cm]  0.101149418
    ## 130            Minimalemptyweight[kg]                 Length[cm]  0.896435250
    ## 131        Permissablegrossweight[kg]                 Length[cm]  0.905896036
    ## 132           Maximumloadcapacity[kg]                 Length[cm]  0.677090091
    ## 133                     Numberofseats                 Length[cm]  0.417210653
    ## 134                     Numberofdoors                 Length[cm]  0.171263315
    ## 135                      Tiresize[in]                 Length[cm]  0.746717048
    ## 136                 Maximumspeed[kph]                 Length[cm]  0.763614974
    ## 137              Bootcapacity(VDA)[l]                 Length[cm]  0.841849339
    ## 138           Acceleration0-100kph[s]                 Length[cm] -0.734826149
    ## 139        MaximumDCchargingpower[kW]                 Length[cm]  0.763375405
    ## 140 mean-Energyconsumption[kWh/100km]                 Length[cm]  0.721075543
    ## 141          Minimalprice(gross)[PLN]                  Width[cm]  0.476217707
    ## 142                   Enginepower[KM]                  Width[cm]  0.471872177
    ## 143                 Maximumtorque[Nm]                  Width[cm]  0.457055008
    ## 144              Batterycapacity[kWh]                  Width[cm]  0.524706737
    ## 145                   Range(WLTP)[km]                  Width[cm]  0.336690121
    ## 146                     Wheelbase[cm]                  Width[cm]  0.488064510
    ## 147                        Length[cm]                  Width[cm]  0.543547107
    ## 149                        Height[cm]                  Width[cm]  0.076035414
    ## 150            Minimalemptyweight[kg]                  Width[cm]  0.524596858
    ## 151        Permissablegrossweight[kg]                  Width[cm]  0.536678981
    ## 152           Maximumloadcapacity[kg]                  Width[cm]  0.402050200
    ## 153                     Numberofseats                  Width[cm]  0.232621034
    ## 154                     Numberofdoors                  Width[cm]  0.044356345
    ## 155                      Tiresize[in]                  Width[cm]  0.500142514
    ## 156                 Maximumspeed[kph]                  Width[cm]  0.474717587
    ## 157              Bootcapacity(VDA)[l]                  Width[cm]  0.514082001
    ## 158           Acceleration0-100kph[s]                  Width[cm] -0.463559152
    ## 159        MaximumDCchargingpower[kW]                  Width[cm]  0.491598065
    ## 160 mean-Energyconsumption[kWh/100km]                  Width[cm]  0.446869118
    ## 161          Minimalprice(gross)[PLN]                 Height[cm] -0.235877258
    ## 162                   Enginepower[KM]                 Height[cm] -0.193816903
    ## 163                 Maximumtorque[Nm]                 Height[cm] -0.092389876
    ## 164              Batterycapacity[kWh]                 Height[cm]  0.041634232
    ## 165                   Range(WLTP)[km]                 Height[cm] -0.011242703
    ## 166                     Wheelbase[cm]                 Height[cm]  0.328453438
    ## 167                        Length[cm]                 Height[cm]  0.101149418
    ## 168                         Width[cm]                 Height[cm]  0.076035414
    ## 170            Minimalemptyweight[kg]                 Height[cm]  0.202155042
    ## 171        Permissablegrossweight[kg]                 Height[cm]  0.216256012
    ## 172           Maximumloadcapacity[kg]                 Height[cm]  0.414319101
    ## 173                     Numberofseats                 Height[cm]  0.619122086
    ## 174                     Numberofdoors                 Height[cm]  0.463129238
    ## 175                      Tiresize[in]                 Height[cm]  0.068011162
    ## 176                 Maximumspeed[kph]                 Height[cm] -0.330497932
    ## 177              Bootcapacity(VDA)[l]                 Height[cm]  0.402828468
    ## 178           Acceleration0-100kph[s]                 Height[cm]  0.303308018
    ## 179        MaximumDCchargingpower[kW]                 Height[cm] -0.285687427
    ## 180 mean-Energyconsumption[kWh/100km]                 Height[cm]  0.132858385
    ## 181          Minimalprice(gross)[PLN]     Minimalemptyweight[kg]  0.786707488
    ## 182                   Enginepower[KM]     Minimalemptyweight[kg]  0.855334110
    ## 183                 Maximumtorque[Nm]     Minimalemptyweight[kg]  0.858799330
    ## 184              Batterycapacity[kWh]     Minimalemptyweight[kg]  0.920965051
    ## 185                   Range(WLTP)[km]     Minimalemptyweight[kg]  0.591383801
    ## 186                     Wheelbase[cm]     Minimalemptyweight[kg]  0.828073460
    ## 187                        Length[cm]     Minimalemptyweight[kg]  0.896435250
    ## 188                         Width[cm]     Minimalemptyweight[kg]  0.524596858
    ## 189                        Height[cm]     Minimalemptyweight[kg]  0.202155042
    ## 191        Permissablegrossweight[kg]     Minimalemptyweight[kg]  0.979724998
    ## 192           Maximumloadcapacity[kg]     Minimalemptyweight[kg]  0.613161895
    ## 193                     Numberofseats     Minimalemptyweight[kg]  0.263144144
    ## 194                     Numberofdoors     Minimalemptyweight[kg]  0.065792365
    ## 195                      Tiresize[in]     Minimalemptyweight[kg]  0.808486092
    ## 196                 Maximumspeed[kph]     Minimalemptyweight[kg]  0.780216240
    ## 197              Bootcapacity(VDA)[l]     Minimalemptyweight[kg]  0.882018345
    ## 198           Acceleration0-100kph[s]     Minimalemptyweight[kg] -0.764068837
    ## 199        MaximumDCchargingpower[kW]     Minimalemptyweight[kg]  0.751934940
    ## 200 mean-Energyconsumption[kWh/100km]     Minimalemptyweight[kg]  0.861909523
    ## 201          Minimalprice(gross)[PLN] Permissablegrossweight[kg]  0.767760358
    ## 202                   Enginepower[KM] Permissablegrossweight[kg]  0.815072787
    ## 203                 Maximumtorque[Nm] Permissablegrossweight[kg]  0.803782763
    ## 204              Batterycapacity[kWh] Permissablegrossweight[kg]  0.885011107
    ## 205                   Range(WLTP)[km] Permissablegrossweight[kg]  0.549737187
    ## 206                     Wheelbase[cm] Permissablegrossweight[kg]  0.867939800
    ## 207                        Length[cm] Permissablegrossweight[kg]  0.905896036
    ## 208                         Width[cm] Permissablegrossweight[kg]  0.536678981
    ## 209                        Height[cm] Permissablegrossweight[kg]  0.216256012
    ## 210            Minimalemptyweight[kg] Permissablegrossweight[kg]  0.979724998
    ## 212           Maximumloadcapacity[kg] Permissablegrossweight[kg]  0.709318405
    ## 213                     Numberofseats Permissablegrossweight[kg]  0.329616122
    ## 214                     Numberofdoors Permissablegrossweight[kg]  0.041930925
    ## 215                      Tiresize[in] Permissablegrossweight[kg]  0.749737656
    ## 216                 Maximumspeed[kph] Permissablegrossweight[kg]  0.750499748
    ## 217              Bootcapacity(VDA)[l] Permissablegrossweight[kg]  0.906140411
    ## 218           Acceleration0-100kph[s] Permissablegrossweight[kg] -0.704823356
    ## 219        MaximumDCchargingpower[kW] Permissablegrossweight[kg]  0.745513493
    ## 220 mean-Energyconsumption[kWh/100km] Permissablegrossweight[kg]  0.874771010
    ## 221          Minimalprice(gross)[PLN]    Maximumloadcapacity[kg]  0.475068582
    ## 222                   Enginepower[KM]    Maximumloadcapacity[kg]  0.432707985
    ## 223                 Maximumtorque[Nm]    Maximumloadcapacity[kg]  0.390599129
    ## 224              Batterycapacity[kWh]    Maximumloadcapacity[kg]  0.507539014
    ## 225                   Range(WLTP)[km]    Maximumloadcapacity[kg]  0.273642010
    ## 226                     Wheelbase[cm]    Maximumloadcapacity[kg]  0.810573337
    ## 227                        Length[cm]    Maximumloadcapacity[kg]  0.677090091
    ## 228                         Width[cm]    Maximumloadcapacity[kg]  0.402050200
    ## 229                        Height[cm]    Maximumloadcapacity[kg]  0.414319101
    ## 230            Minimalemptyweight[kg]    Maximumloadcapacity[kg]  0.613161895
    ## 231        Permissablegrossweight[kg]    Maximumloadcapacity[kg]  0.709318405
    ## 233                     Numberofseats    Maximumloadcapacity[kg]  0.536034513
    ## 234                     Numberofdoors    Maximumloadcapacity[kg] -0.002314982
    ## 235                      Tiresize[in]    Maximumloadcapacity[kg]  0.419874021
    ## 236                 Maximumspeed[kph]    Maximumloadcapacity[kg]  0.447107139
    ## 237              Bootcapacity(VDA)[l]    Maximumloadcapacity[kg]  0.728206393
    ## 238           Acceleration0-100kph[s]    Maximumloadcapacity[kg] -0.255273354
    ## 239        MaximumDCchargingpower[kW]    Maximumloadcapacity[kg]  0.509866271
    ## 240 mean-Energyconsumption[kWh/100km]    Maximumloadcapacity[kg]  0.650026546
    ## 241          Minimalprice(gross)[PLN]              Numberofseats -0.090688436
    ## 242                   Enginepower[KM]              Numberofseats -0.087453650
    ## 243                 Maximumtorque[Nm]              Numberofseats -0.001363035
    ## 244              Batterycapacity[kWh]              Numberofseats  0.167248289
    ## 245                   Range(WLTP)[km]              Numberofseats  0.130506408
    ## 246                     Wheelbase[cm]              Numberofseats  0.625667199
    ## 247                        Length[cm]              Numberofseats  0.417210653
    ## 248                         Width[cm]              Numberofseats  0.232621034
    ## 249                        Height[cm]              Numberofseats  0.619122086
    ## 250            Minimalemptyweight[kg]              Numberofseats  0.263144144
    ## 251        Permissablegrossweight[kg]              Numberofseats  0.329616122
    ## 252           Maximumloadcapacity[kg]              Numberofseats  0.536034513
    ## 254                     Numberofdoors              Numberofseats  0.616216585
    ## 255                      Tiresize[in]              Numberofseats  0.024790506
    ## 256                 Maximumspeed[kph]              Numberofseats -0.160005945
    ## 257              Bootcapacity(VDA)[l]              Numberofseats  0.407679455
    ## 258           Acceleration0-100kph[s]              Numberofseats  0.122769541
    ## 259        MaximumDCchargingpower[kW]              Numberofseats  0.021871644
    ## 260 mean-Energyconsumption[kWh/100km]              Numberofseats  0.183648113
    ## 261          Minimalprice(gross)[PLN]              Numberofdoors -0.306692899
    ## 262                   Enginepower[KM]              Numberofdoors -0.260890205
    ## 263                 Maximumtorque[Nm]              Numberofdoors -0.133803691
    ## 264              Batterycapacity[kWh]              Numberofdoors  0.038005880
    ## 265                   Range(WLTP)[km]              Numberofdoors  0.152500139
    ## 266                     Wheelbase[cm]              Numberofdoors  0.264680729
    ## 267                        Length[cm]              Numberofdoors  0.171263315
    ## 268                         Width[cm]              Numberofdoors  0.044356345
    ## 269                        Height[cm]              Numberofdoors  0.463129238
    ## 270            Minimalemptyweight[kg]              Numberofdoors  0.065792365
    ## 271        Permissablegrossweight[kg]              Numberofdoors  0.041930925
    ## 272           Maximumloadcapacity[kg]              Numberofdoors -0.002314982
    ## 273                     Numberofseats              Numberofdoors  0.616216585
    ## 275                      Tiresize[in]              Numberofdoors -0.016257241
    ## 276                 Maximumspeed[kph]              Numberofdoors -0.302964824
    ## 277              Bootcapacity(VDA)[l]              Numberofdoors  0.188547338
    ## 278           Acceleration0-100kph[s]              Numberofdoors  0.212901102
    ## 279        MaximumDCchargingpower[kW]              Numberofdoors -0.236286284
    ## 280 mean-Energyconsumption[kWh/100km]              Numberofdoors -0.172312792
    ## 281          Minimalprice(gross)[PLN]               Tiresize[in]  0.750241486
    ## 282                   Enginepower[KM]               Tiresize[in]  0.800977748
    ## 283                 Maximumtorque[Nm]               Tiresize[in]  0.749802241
    ## 284              Batterycapacity[kWh]               Tiresize[in]  0.788210908
    ## 285                   Range(WLTP)[km]               Tiresize[in]  0.618844536
    ## 286                     Wheelbase[cm]               Tiresize[in]  0.643339062
    ## 287                        Length[cm]               Tiresize[in]  0.746717048
    ## 288                         Width[cm]               Tiresize[in]  0.500142514
    ## 289                        Height[cm]               Tiresize[in]  0.068011162
    ## 290            Minimalemptyweight[kg]               Tiresize[in]  0.808486092
    ## 291        Permissablegrossweight[kg]               Tiresize[in]  0.749737656
    ## 292           Maximumloadcapacity[kg]               Tiresize[in]  0.419874021
    ## 293                     Numberofseats               Tiresize[in]  0.024790506
    ## 294                     Numberofdoors               Tiresize[in] -0.016257241
    ## 296                 Maximumspeed[kph]               Tiresize[in]  0.769335697
    ## 297              Bootcapacity(VDA)[l]               Tiresize[in]  0.639452615
    ## 298           Acceleration0-100kph[s]               Tiresize[in] -0.827448445
    ## 299        MaximumDCchargingpower[kW]               Tiresize[in]  0.670548866
    ## 300 mean-Energyconsumption[kWh/100km]               Tiresize[in]  0.577979423
    ## 301          Minimalprice(gross)[PLN]          Maximumspeed[kph]  0.943811567
    ## 302                   Enginepower[KM]          Maximumspeed[kph]  0.951682772
    ## 303                 Maximumtorque[Nm]          Maximumspeed[kph]  0.877643876
    ## 304              Batterycapacity[kWh]          Maximumspeed[kph]  0.823871149
    ## 305                   Range(WLTP)[km]          Maximumspeed[kph]  0.561107675
    ## 306                     Wheelbase[cm]          Maximumspeed[kph]  0.595937406
    ## 307                        Length[cm]          Maximumspeed[kph]  0.763614974
    ## 308                         Width[cm]          Maximumspeed[kph]  0.474717587
    ## 309                        Height[cm]          Maximumspeed[kph] -0.330497932
    ## 310            Minimalemptyweight[kg]          Maximumspeed[kph]  0.780216240
    ## 311        Permissablegrossweight[kg]          Maximumspeed[kph]  0.750499748
    ## 312           Maximumloadcapacity[kg]          Maximumspeed[kph]  0.447107139
    ## 313                     Numberofseats          Maximumspeed[kph] -0.160005945
    ## 314                     Numberofdoors          Maximumspeed[kph] -0.302964824
    ## 315                      Tiresize[in]          Maximumspeed[kph]  0.769335697
    ## 317              Bootcapacity(VDA)[l]          Maximumspeed[kph]  0.575806084
    ## 318           Acceleration0-100kph[s]          Maximumspeed[kph] -0.887069110
    ## 319        MaximumDCchargingpower[kW]          Maximumspeed[kph]  0.920975113
    ## 320 mean-Energyconsumption[kWh/100km]          Maximumspeed[kph]  0.730469647
    ## 321          Minimalprice(gross)[PLN]       Bootcapacity(VDA)[l]  0.575947547
    ## 322                   Enginepower[KM]       Bootcapacity(VDA)[l]  0.644273618
    ## 323                 Maximumtorque[Nm]       Bootcapacity(VDA)[l]  0.676182162
    ## 324              Batterycapacity[kWh]       Bootcapacity(VDA)[l]  0.794238773
    ## 325                   Range(WLTP)[km]       Bootcapacity(VDA)[l]  0.508850248
    ## 326                     Wheelbase[cm]       Bootcapacity(VDA)[l]  0.844244547
    ## 327                        Length[cm]       Bootcapacity(VDA)[l]  0.841849339
    ## 328                         Width[cm]       Bootcapacity(VDA)[l]  0.514082001
    ## 329                        Height[cm]       Bootcapacity(VDA)[l]  0.402828468
    ## 330            Minimalemptyweight[kg]       Bootcapacity(VDA)[l]  0.882018345
    ## 331        Permissablegrossweight[kg]       Bootcapacity(VDA)[l]  0.906140411
    ## 332           Maximumloadcapacity[kg]       Bootcapacity(VDA)[l]  0.728206393
    ## 333                     Numberofseats       Bootcapacity(VDA)[l]  0.407679455
    ## 334                     Numberofdoors       Bootcapacity(VDA)[l]  0.188547338
    ## 335                      Tiresize[in]       Bootcapacity(VDA)[l]  0.639452615
    ## 336                 Maximumspeed[kph]       Bootcapacity(VDA)[l]  0.575806084
    ## 338           Acceleration0-100kph[s]       Bootcapacity(VDA)[l] -0.524969320
    ## 339        MaximumDCchargingpower[kW]       Bootcapacity(VDA)[l]  0.534582199
    ## 340 mean-Energyconsumption[kWh/100km]       Bootcapacity(VDA)[l]  0.772115661
    ## 341          Minimalprice(gross)[PLN]    Acceleration0-100kph[s] -0.810990436
    ## 342                   Enginepower[KM]    Acceleration0-100kph[s] -0.896693568
    ## 343                 Maximumtorque[Nm]    Acceleration0-100kph[s] -0.832416924
    ## 344              Batterycapacity[kWh]    Acceleration0-100kph[s] -0.819402510
    ## 345                   Range(WLTP)[km]    Acceleration0-100kph[s] -0.627631213
    ## 346                     Wheelbase[cm]    Acceleration0-100kph[s] -0.537228280
    ## 347                        Length[cm]    Acceleration0-100kph[s] -0.734826149
    ## 348                         Width[cm]    Acceleration0-100kph[s] -0.463559152
    ## 349                        Height[cm]    Acceleration0-100kph[s]  0.303308018
    ## 350            Minimalemptyweight[kg]    Acceleration0-100kph[s] -0.764068837
    ## 351        Permissablegrossweight[kg]    Acceleration0-100kph[s] -0.704823356
    ## 352           Maximumloadcapacity[kg]    Acceleration0-100kph[s] -0.255273354
    ## 353                     Numberofseats    Acceleration0-100kph[s]  0.122769541
    ## 354                     Numberofdoors    Acceleration0-100kph[s]  0.212901102
    ## 355                      Tiresize[in]    Acceleration0-100kph[s] -0.827448445
    ## 356                 Maximumspeed[kph]    Acceleration0-100kph[s] -0.887069110
    ## 357              Bootcapacity(VDA)[l]    Acceleration0-100kph[s] -0.524969320
    ## 359        MaximumDCchargingpower[kW]    Acceleration0-100kph[s] -0.772705711
    ## 360 mean-Energyconsumption[kWh/100km]    Acceleration0-100kph[s] -0.626844768
    ## 361          Minimalprice(gross)[PLN] MaximumDCchargingpower[kW]  0.887719302
    ## 362                   Enginepower[KM] MaximumDCchargingpower[kW]  0.854077056
    ## 363                 Maximumtorque[Nm] MaximumDCchargingpower[kW]  0.770809047
    ## 364              Batterycapacity[kWh] MaximumDCchargingpower[kW]  0.788901752
    ## 365                   Range(WLTP)[km] MaximumDCchargingpower[kW]  0.551216206
    ## 366                     Wheelbase[cm] MaximumDCchargingpower[kW]  0.627443514
    ## 367                        Length[cm] MaximumDCchargingpower[kW]  0.763375405
    ## 368                         Width[cm] MaximumDCchargingpower[kW]  0.491598065
    ## 369                        Height[cm] MaximumDCchargingpower[kW] -0.285687427
    ## 370            Minimalemptyweight[kg] MaximumDCchargingpower[kW]  0.751934940
    ## 371        Permissablegrossweight[kg] MaximumDCchargingpower[kW]  0.745513493
    ## 372           Maximumloadcapacity[kg] MaximumDCchargingpower[kW]  0.509866271
    ## 373                     Numberofseats MaximumDCchargingpower[kW]  0.021871644
    ## 374                     Numberofdoors MaximumDCchargingpower[kW] -0.236286284
    ## 375                      Tiresize[in] MaximumDCchargingpower[kW]  0.670548866
    ## 376                 Maximumspeed[kph] MaximumDCchargingpower[kW]  0.920975113
    ## 377              Bootcapacity(VDA)[l] MaximumDCchargingpower[kW]  0.534582199
    ## 378           Acceleration0-100kph[s] MaximumDCchargingpower[kW] -0.772705711
    ## 380 mean-Energyconsumption[kWh/100km] MaximumDCchargingpower[kW]  0.709328384

# 

##### A variável “Permissable gross weight \[kg\]” tem forte correlação com a variável Minimal empty weight \[kg\]” e será removida.

##### A variável “Engine power \[KM\]” tem correlação com as variáveis “Maximum torque \[Nm\]” e “Maximum speed \[kph\]”, sendo que a primeira influencia nas outras e apenas “Engine power \[KM\]” será mantida.

# 

##### Correlação das variáveis preditoras com a variável alvo:

``` r
corr_matrix[corr_matrix$Var1 != "mean-Energyconsumption[kWh/100km]" & 
                   corr_matrix$Var2 == "mean-Energyconsumption[kWh/100km]", ]
```

    ##                           Var1                              Var2      value
    ## 381   Minimalprice(gross)[PLN] mean-Energyconsumption[kWh/100km]  0.7994474
    ## 382            Enginepower[KM] mean-Energyconsumption[kWh/100km]  0.8246484
    ## 383          Maximumtorque[Nm] mean-Energyconsumption[kWh/100km]  0.8276390
    ## 384       Batterycapacity[kWh] mean-Energyconsumption[kWh/100km]  0.7578657
    ## 385            Range(WLTP)[km] mean-Energyconsumption[kWh/100km]  0.2744891
    ## 386              Wheelbase[cm] mean-Energyconsumption[kWh/100km]  0.7164781
    ## 387                 Length[cm] mean-Energyconsumption[kWh/100km]  0.7210755
    ## 388                  Width[cm] mean-Energyconsumption[kWh/100km]  0.4468691
    ## 389                 Height[cm] mean-Energyconsumption[kWh/100km]  0.1328584
    ## 390     Minimalemptyweight[kg] mean-Energyconsumption[kWh/100km]  0.8619095
    ## 391 Permissablegrossweight[kg] mean-Energyconsumption[kWh/100km]  0.8747710
    ## 392    Maximumloadcapacity[kg] mean-Energyconsumption[kWh/100km]  0.6500265
    ## 393              Numberofseats mean-Energyconsumption[kWh/100km]  0.1836481
    ## 394              Numberofdoors mean-Energyconsumption[kWh/100km] -0.1723128
    ## 395               Tiresize[in] mean-Energyconsumption[kWh/100km]  0.5779794
    ## 396          Maximumspeed[kph] mean-Energyconsumption[kWh/100km]  0.7304696
    ## 397       Bootcapacity(VDA)[l] mean-Energyconsumption[kWh/100km]  0.7721157
    ## 398    Acceleration0-100kph[s] mean-Energyconsumption[kWh/100km] -0.6268448
    ## 399 MaximumDCchargingpower[kW] mean-Energyconsumption[kWh/100km]  0.7093284

# 

##### Serão consideradas para o modelo as variáveis que possuem correlação acima de 0.5 ou abaixo de -0.5 com a variável alvo.

# 

##### As variáveis categóricas serão abandonadas, pois a diferença de marca também pode ser sentida na diferença de preço para parâmetros similares.

``` r
val_to_drop <- c("Minimalemptyweight[kg]", "Maximumtorque[Nm]", 
                 "Maximumspeed[kph]")

var_selec <- corr_matrix[((corr_matrix$value > 0.5 | corr_matrix$value < -0.5) & 
                           (corr_matrix$Var2 == "mean-Energyconsumption[kWh/100km]") &
                            !corr_matrix$Var1 %in% val_to_drop), 1] %>% as.character
```

# 

##### Aplicando normalização no dataset com as variáveis selecionadas:

``` r
df2_num_scale <- scale_data_frame(df = df2_num, col_list = var_selec)
```

# 

##### Dividindo em dados de treino e teste na proporção 75/25:

``` r
set.seed(1234)

indexes <- sample(1:nrow(df2_num_scale), size = 0.75 * nrow(df2_num_scale))
x_train <- df2_num_scale[indexes, 1:(length(df2_num_scale)-1)]
y_train <- df2_num_scale[indexes, length(df2_num_scale)]
x_test <- df2_num_scale[-indexes, 1:(length(df2_num_scale)-1)]
y_test <- df2_num_scale[-indexes, length(df2_num_scale)]

rm(col_to_drop, corr_matrix, df, df2,  df2_num, df2_num_scale, indexes, val_to_drop, var_selec)
```

# 

##### Treinando os modelos escolhidos com caretEnsemble:

``` r
fitControl= trainControl(method="cv", number = 5, 
                         savePredictions = "final", allowParallel = TRUE)

model_list <- caretList(x = x_train, y = y_train, trControl = fitControl, 
                        methodList = c("lm", "svmRadial", "rf", "bridge", "xgbLinear"), 
                        tuneList = NULL, continue_on_fail = FALSE)
```

    ## Warning in trControlCheck(x = trControl, y = target): indexes not defined in
    ## trControl. Attempting to set them ourselves, so each model in the ensemble will
    ## have the same resampling indexes.

    ## t=100, m=9
    ## t=200, m=11
    ## t=300, m=15
    ## t=400, m=7
    ## t=500, m=10
    ## t=600, m=11
    ## t=700, m=11
    ## t=800, m=7
    ## t=900, m=14
    ## t=100, m=9
    ## t=200, m=16
    ## t=300, m=12
    ## t=400, m=8
    ## t=500, m=9
    ## t=600, m=10
    ## t=700, m=6
    ## t=800, m=8
    ## t=900, m=9
    ## t=100, m=8
    ## t=200, m=6
    ## t=300, m=8
    ## t=400, m=9
    ## t=500, m=8
    ## t=600, m=7
    ## t=700, m=8
    ## t=800, m=10
    ## t=900, m=9
    ## t=100, m=10
    ## t=200, m=11
    ## t=300, m=7
    ## t=400, m=7
    ## t=500, m=13
    ## t=600, m=9
    ## t=700, m=12
    ## t=800, m=8
    ## t=900, m=11
    ## t=100, m=10
    ## t=200, m=10
    ## t=300, m=15
    ## t=400, m=14
    ## t=500, m=9
    ## t=600, m=11
    ## t=700, m=11
    ## t=800, m=8
    ## t=900, m=13
    ## t=100, m=12
    ## t=200, m=12
    ## t=300, m=8
    ## t=400, m=10
    ## t=500, m=8
    ## t=600, m=9
    ## t=700, m=10
    ## t=800, m=11
    ## t=900, m=12

# 

##### Fazendo as predições:

``` r
pred_lm <- predict.train(model_list$lm, newdata = x_test)
pred_svm <- predict.train(model_list$svmRadial, newdata = x_test)
pred_rf <- predict.train(model_list$rf, newdata = x_test)
pred_bridge <- predict.train(model_list$bridge, newdata = x_test)
pred_xgbL <- predict.train(model_list$xgbLinear, newdata = x_test)
```

# 

##### Avaliando os modelos:

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

    ##                       LM       SVM        RF    BRIDGE      XGBL
    ## RMSE_model     0.8946604 0.4364300 0.3495653 0.3837875 0.4124066
    ## RMSE_predict   3.5717255 0.6295264 0.3689778 0.4010967 0.3682480
    ## Corr_Pred_Real 0.1759997 0.8718217 0.9479470 0.9146299 0.9250850

# 

##### De acordo com os resultados da métrica RMSE o melhor modelo foi o Random Forest.

``` r
modelo_final <- model_list$rf
```
