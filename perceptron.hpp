
#pragma once

#include <iostream>
#include <vector>
#include <assert.h>
#include <random>
#include <chrono>

using namespace std;

// perceptron class
class perceptron {

public:

	// constructor
	perceptron(int m=0);

	// Evaluate the perceptron
	int activate(vector<long double> & X);

	// Update the perceptron
	void update(vector<long double> & X, int error, long double learning_rate);

	// train the perceptron
	void train(vector<vector<long double> > & epoch, vector<int> & d, long double learning_rate, int limit);

	// initialize the weights to random 
	void reset_weight_to_random();

	// print the weights
	void show();

private:
	
	// input size, weight's vector and some things to get pseudo random numbers 
	int m;
	vector<long double> weight;
	static default_random_engine generator;
	static uniform_real_distribution<long double> distribution;

};