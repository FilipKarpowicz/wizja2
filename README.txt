Aby wyłączyć rozpoznawanie obrazow za pomoca sieci neuronowej nalezy w pliku main.py (linia 17) ustawic wartosc
zmiennej czy_model z True na False

indeksowanie cech
0. liczba Eulera (EulerNumber) 
1. Pole powierzchnie (Area)
2. Pole powierzchni najmniejszego prostokąta (BoundingBoxArea)
3. Pole powierzchni najmniejszego prostokąta z usuniętymi otworami (FilledArea)
4. Stosunek pola powierchni obiektu od pola powierzchni najmniejszego prostokąta (Extent)
5. Promień koła o powierzchni równej powierzechni obiektu (EquivDiameter)
6. Stosunek pola powierchni obiektu od pola powierzchni jego powłoki wypukłej (Solidity)
7. Pole powierzchni otworów w obiekcie
8. Log momentu Hu1
9. Log momentu Hu2
10. Log momentu Hu3
11. Log momentu Hu4
12. Log momentu Hu5
13. Log momentu Hu6
14. Log momentu Hu7
15. Składowa koloru R/B
16. Składowa koloru G
17. Składowa koloru B/R


Wzór na klasy

k1 = 1 dla kwadratow
k1 = 0 dla kol

k2 = 0 dla koloru zielonego
k2 = 1 dla koloru magenta
k2 = 2 dla koloru szary

k3 = 0 duzy
k3 = 1 sredni
k3 = 2 maly

k1*9+k2*3+k3

0. kolo zielone duze
1. kolo zielone srednie
2. kolo zielone male
3. kolo magenta duze
4. kolo magenta srednie
5. kolo magenta male
6. kolo szare duze
7. kolo szare srednie
9. kolo szare male
10. kwadrat zielony duzy
11. kwadrat zielony sredni
12. kwadrat zielony maly
13. kwadrat magenty duzy
14. kwadrat magenty sredni
15. kwadrat magenty maly
16. kwadrat szary duzy
17. kwadrat szary sredni
19. kwadrat szary maly





