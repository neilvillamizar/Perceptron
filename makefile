
perceptron: main.cpp perceptron.o perceptron_neural_network.o
	g++ -std=c++17 main.cpp perceptron.o perceptron_neural_network.o -o perceptron

perceptron_neural_network.o : perceptron_neural_network.cpp perceptron_neural_network.hpp
	g++ -std=c++17 -c perceptron_neural_network.cpp

perceptron.o: perceptron.cpp perceptron.hpp
	g++ -std=c++17 -c perceptron.cpp

clean:
	rm perceptron
	rm *.o