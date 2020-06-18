
#include "perceptron.hpp"

// constructor
perceptron::perceptron(int m_) : m(m_) {
	reset_weight_to_random();
}

// initialize the weights to random
void perceptron::reset_weight_to_random(){

	weight.resize(m + 1);

	for (long double & w : weight){
		w = distribution(generator);
	}

}

// Evaluate the perceptron
int perceptron::activate(vector<long double> & X){

	assert(m == X.size());
	// In this case the bias is in the last position
	long double x = 0;

	x += weight[m];

	for(int i=0; i<m; i++){
		x += weight[i]*X[i];
	}

	return (x >= 0 ? 1 : 0);

}

// Update the perceptron
void perceptron::update(vector<long double> & X, int error, long double learning_rate){

	assert(m == X.size());

	if(!error) return;
	// In this case the bias is in the last position
	weight[m] += learning_rate * (long double)error;

	for(int i = 0; i <m; i++){
		weight[i] += learning_rate * (long double)error * X[i];
	}

}

// train the perceptron
void perceptron::train(vector<vector<long double> > & epoch, vector<int> & d, long double learning_rate, int limit){

	reset_weight_to_random();

	bool conv = false;
	int it = 0;         

	while(!conv && it < limit){
		
		int i = 0;
		conv = true;

		for(auto & X : epoch){

			int y = activate(X);
			int e = d[i] - y;
			update(X, e, learning_rate);
			if(e) conv = false;
			i++;

		}

		it++;
	}

}

// print the weights
void perceptron::show(){
	for(auto w:weight){
		cout << w << " ";
	}
	cout << endl;
}

// initialize an uniform distribution in range (-0.05,0.05)
uniform_real_distribution<long double> perceptron::distribution(-0.05,0.05);

// initialize with a seed a psudo-random generator 
default_random_engine perceptron::generator(chrono::system_clock::now().time_since_epoch().count());

