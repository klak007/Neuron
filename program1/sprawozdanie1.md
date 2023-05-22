## Perceptron oraz Adaline
1. Opis problemu
Zadaniem była implementacja perceptronu i nauczenie go poprawnej klasyfikacji gatunków irysów, dla zbioru danych Iris.data. Neuron należało zaimplementować petodą perceptronu oraz Adaline, a następnie porównać klasyfikowanie na podstawie dwóch i trzech klas.
2. Przebeg zadania
   1. Perceptron 
   
        ![](schematPerceptron.png)

        Dla zbioru Z = {(x<sup>(1)</sup>,y<sup>(1)</sup>),...,(x<sup>(n)</sup>,y<sup>(n)</sup>)}

        od ustalonej liczby n<sub>epoch</sub> nalezy iterować po zbiorze Z

        dla i=1,...,N nalezy obliczyć wagi,

        error(i)=y<sup>(i)</sup>-o(x<sup>(i)</sup>)

        $\Delta$