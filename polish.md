# Eksploracja Danych w R

## Przegląd danych
Charakterystyka: | Rozmiar: | Ilość atrybutów: | Charakterystyka atrybutów: | Powiązane zadania:
--- | --- | --- | --- | ---
wielowymiarowe | 5 x 1372 | 5 | liczby rzeczywiste | klasyfikacja (ang. classification)

Dane zostały wydobyte ze zdjęć oryginalnych i podrobionych banknotów. Do digitalizacji użyto kamery przemysłowej zwykle używanej do kontroli wydruku. Zdjęcia mają rozmiar 400 x 400 pikseli, są w skali szarości o rozdzielczości około 660 dpi. Narzędzie Wavelet Transform zostało użyte do wydobycia atrybutów ze zdjęć.

**Atrybuty**:
1. wariancja (ang. variance),
2. współczynnik skośności (ang. skewness),
3. kurtoza (ang. curtosis),
4. entropia (ang. entropy),
5. klasa (ang. class).

**Źródło**: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

## Statystyki ogólne
Ładowanie pakietów:
```R
library(caret)
library(data.table)
library(dplyr)
library(PerformanceAnalytics)
library(rpart.plot)
```

Ładowanie zbioru danych:
```R
url = 'http://bit.ly/banknote-auth'
df = data.frame(fread(url))
names(df) = c('variance', 'skewness', 'curtosis', 'entropy', 'class')
```

Sprawdzimy strukturę danych:
```R
str(df)
```

Wynik:
```
'data.frame':	1372 obs. of  5 variables:
 $ variance: num  3.622 4.546 3.866 3.457 0.329 ...
 $ skewness: num  8.67 8.17 -2.64 9.52 -4.46 ...
 $ curtosis: num  -2.81 -2.46 1.92 -4.01 4.57 ...
 $ entropy : num  -0.447 -1.462 0.106 -3.594 -0.989 ...
 $ class   : int  0 0 0 0 0 0 0 0 0 0 ...
```

Wyświetlimy pierwsze pięć wierszy:
```R
head(df, 5)
```

<table>
<thead><tr><th scope=col>variance</th><th scope=col>skewness</th><th scope=col>curtosis</th><th scope=col>entropy</th><th scope=col>class</th></tr></thead>
<tbody>
	<tr><td>3.62160 </td><td> 8.6661 </td><td>-2.8073 </td><td>-0.44699</td><td>0       </td></tr>
	<tr><td>4.54590 </td><td> 8.1674 </td><td>-2.4586 </td><td>-1.46210</td><td>0       </td></tr>
	<tr><td>3.86600 </td><td>-2.6383 </td><td> 1.9242 </td><td> 0.10645</td><td>0       </td></tr>
	<tr><td>3.45660 </td><td> 9.5228 </td><td>-4.0112 </td><td>-3.59440</td><td>0       </td></tr>
	<tr><td>0.32924 </td><td>-4.4552 </td><td> 4.5718 </td><td>-0.98880</td><td>0       </td></tr>
</tbody>
</table>
