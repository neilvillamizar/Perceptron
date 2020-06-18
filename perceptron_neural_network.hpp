#pragma once

#include <iostream>
#include <vector>

#include "perceptron.hpp"

using namespace std;

// Class perceptron_neural_network that represents a neural network of perceptrons
class perceptron_neural_network {

public:

	// Constructor
	perceptron_neural_network(int n, int m);

	// Training algorithm for the neural network with some data set
	void train_neural_network( 
		vector<vector<long double> > & train_data, vector<vector<int> > & results, long double learning_rate, int limit=1);

	// Classify an image
	vector<int> classify(vector<long double> & X);

	// test the network with some data set
	pair<int,int> test_neural_network(vector<vector<long double> > & test_data, vector<vector<int> > & results);

private:

	// Number of neurons and input
	int N, M;

	// vector of perceptrons
	vector<perceptron > neural_network;

};
