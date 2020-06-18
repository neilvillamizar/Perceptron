


#include "perceptron_neural_network.hpp"

// Constructor
perceptron_neural_network::perceptron_neural_network(int n, int m) : N(n), M(m) {

	for(int i=0; i < N; i++){
		perceptron perc(M);
		neural_network.push_back(perc);
	}

}

// Training algorithm for the neural network with some data set
void perceptron_neural_network::train_neural_network( 
		vector<vector<long double> > & train_data, vector<vector<int> > & results, long double learning_rate, int limit){

	int perro = 0;

	for(int ep=0; ep<limit; ep++){
		int it = 0;

		for(auto & X: train_data){
			for(int digit=0; digit<N; digit++){
				int y = neural_network[digit].activate(X);
				int e = results[it][digit] - y;
				neural_network[digit].update(X, e, learning_rate);
			}
			it++;
		}
	}

}

// Classify an image
vector<int> perceptron_neural_network::classify(vector<long double> & X){
	
	vector<int> y(N);

	for(int i=0; i<N; i++){
		y[i] = neural_network[i].activate(X);
	}

	return y;

}

// test the network with some data set
pair<int,int> perceptron_neural_network::test_neural_network( 
												vector<vector<long double> > & test_data, vector<vector<int> > & results){

	int success = 0, it = 0;
	int total = test_data.size();

	for(auto X: test_data){
		vector<int> v1, v2;
		v1 = classify(X);
		v2 = results[it];

		if (v1 == v2)
			success++;
		it++;
	}

	cout << "total number of tests: " << total << endl;
	cout << "number of successful classifications: " << success << endl;
	cout << "number of wrong classifications: " << (total - success) << endl;

	return {total, success};

}
