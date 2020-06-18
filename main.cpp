
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <stdio.h>

#include "perceptron_neural_network.hpp"

#define N_PIXELS 784						// Number of pixels in one image
#define N_PERCEP 10							// number of perceptrons in one perceptron neural network
#define ITERATIONS 50						// The number of iterations of the training algorithm
const long double ETHA1 = 0.001;			// ETHA1, ETHA2 and ETHA3 are learning rates for the training algorithm
const long double ETHA2 = 0.01;
const long double ETHA3 = 0.1;

using namespace std;


// Function to read data from a CSV file 
void readCSV_data(FILE * fd, vector<vector<long double> > & train_data, vector<vector<int> > & results, int flag=0){

	int data;
	// rnd is a pseudo random generator used in the random shuffle
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine rnd(seed);
	vector< pair<vector<long double>, vector<int>> > lines;

	// loop for reading the data
	while(fscanf(fd, "%d", &data)){

		vector<long double> pixels;
		// store the answer
		vector<int> aux_v(N_PERCEP, 0);
		aux_v[data] = 1;

		// Get the pixels
		for(int i=0; i < N_PIXELS; i++){
			fscanf(fd, ",%d", &data);
			pixels.push_back((long double)data/255.0);
		}
		
		// store the pixles and the corresponding answer as a pair
		// to maintaing the relation after the shuffle
		lines.push_back({pixels, aux_v});

	}

	// If is the training data, shuffle
	if(flag) shuffle(lines.begin(), lines.end(), rnd);

	// Store the data in two vectors
	for(auto & pr : lines){
		results.push_back(pr.second);
		train_data.push_back(pr.first);
	}

}

// calculate the percentage of x, if the total is t 
double percentage(double t, double x){
	return 100.0*x/((double)t);
}

// Main function
int main(int argc, char const *argv[]) {
	
	// vectors to store the data used to train and test
	vector<vector<long double> > mnist_train, mnist_test;
	vector<vector<int> > mnist_results, mnist_test_results;
	pair<int, int> st;

	// the 3 neural networks
	perceptron_neural_network neural_netw1(N_PERCEP, N_PIXELS);
	perceptron_neural_network neural_netw2(N_PERCEP, N_PIXELS);
	perceptron_neural_network neural_netw3(N_PERCEP, N_PIXELS);

	// open, read and close the file with the training data
	FILE * fd = fopen("mnist_train.csv", "r");
	readCSV_data(fd, mnist_train, mnist_results, 1);
	fclose(fd);
	
	// training process
	cout << "Training perceptron with learning rate of " << ETHA1 << ":\n";
	neural_netw1.train_neural_network(mnist_train, mnist_results, ETHA1, ITERATIONS);
	cout << "Training perceptron with learning rate of " << ETHA2 << ":\n";
	neural_netw2.train_neural_network(mnist_train, mnist_results, ETHA2, ITERATIONS);
	cout << "Training perceptron with learning rate of " << ETHA3 << ":\n";
	neural_netw3.train_neural_network(mnist_train, mnist_results, ETHA3, ITERATIONS);

	mnist_train.clear();
	mnist_results.clear();

	// open, read and close the file with the testing data
	fd = fopen("mnist_test.csv", "r");
	readCSV_data(fd, mnist_test, mnist_test_results);
	fclose(fd);

	// Testing process
	cout << "Using perceptron with learning rate of " << ETHA1 << ":\n";
	st = neural_netw1.test_neural_network(mnist_test, mnist_test_results);
	cout << "The percentage of successful classifications is: " << percentage(st.first, st.second) << endl;

	cout << "Using perceptron with learning rate of " << ETHA2 << ":\n";
	st = neural_netw2.test_neural_network(mnist_test, mnist_test_results);
	cout << "The percentage of successful classifications is: " << percentage(st.first, st.second) << endl;

	cout << "Using perceptron with learning rate of " << ETHA3 << ":\n";
	st = neural_netw3.test_neural_network(mnist_test, mnist_test_results);
	cout << "The percentage of successful classifications is: " << percentage(st.first, st.second) << endl;

	
	return 0;
}