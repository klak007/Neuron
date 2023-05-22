## Klasyfikacja wieloklasowa
1. Opis problemu
   Zadaniem było przeprowadzenie klasyfikacji wieloklasowej przy pomocy wielowarstwowego perceptronu. Przeprowadzono to na trzech zestawach danych: Iris, liczb pisanych odręcznie (MNIST) oraz datasetu jakości win. Należało również uruchomić funkcję softmax na wyjściu sieci dla klasyfikacji wieloklasowej, a następnie dobrać parametry dla neuronu za pomoć funkcji GridSearchCV, szukającej hiper-parametrów.
2. Perceptron wielowarstwowy
   ![](schematPerceptronWielowarstwowy.png)
   Wejściem perceptronu jest wektor 28 * 28, z kodowaniem liter $\\$
   Warstwy ukryte - to jedna z funkcji aktywacji $\\$
   Na wyjściu zastosowana funkcja softmax.
   $\\$
   Funkcja softmax:
   $ \sigma (z)_i $