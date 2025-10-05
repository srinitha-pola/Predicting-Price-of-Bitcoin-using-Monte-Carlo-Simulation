#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

#define CUDA_CHECK(err) do {     if (err != cudaSuccess) {         cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << endl;         exit(1);     } } while(0)

vector<string> split(const string& s, char delimiter) {
    vector<string> tokens;
    string token;
    stringstream ss(s);
    while (getline(ss, token, delimiter)) tokens.push_back(token);
    return tokens;
}

vector<double> readCSV(const string& filename) {
    ifstream file(filename);
    vector<double> prices;
    string line;

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return prices;
    }

    for (int i = 0; i < 3 && getline(file, line); ++i) {}

    while (getline(file, line)) {
        vector<string> tokens = split(line, ',');
        if (tokens.size() < 2) continue;
        try {
            prices.push_back(stod(tokens[1]));
        } catch (...) {
            continue;
        }
    }
    return prices;
}

__global__ void monteCarloKernel(double* finalPrices, double lastPrice, double mu, double sigma, int days, int simulations, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= simulations) return;

    curandState state;
    curand_init(seed + idx, 0, 0, &state); 

    double price = lastPrice;
    double dt = 1.0;
    for (int d = 0; d < days; d++) {
        double z = curand_normal_double(&state); 
        double drift = (mu - 0.5 * sigma * sigma) * dt;
        double shock = sigma * sqrt(dt) * z;
        price *= exp(drift + shock);
    }
    finalPrices[idx] = price;
}

int main() {
    clock_t start_time = clock();

    string filename = "aapl_stock_data.csv";
    vector<double> prices = readCSV(filename);

    if (prices.size() < 60) {
        cerr << "Not enough price data! Need at least 60 data points." << endl;
        return 1;
    }

    vector<double> logReturns;
    for (size_t i = 1; i < prices.size(); i++) {
        if (prices[i - 1] != 0) {
            double r = log(prices[i] / prices[i - 1]);
            logReturns.push_back(r);
        }
    }

    if (logReturns.empty()) {
        cerr << "No valid log returns calculated!" << endl;
        return 1;
    }

    double mean_r = accumulate(logReturns.begin(), logReturns.end(), 0.0) / logReturns.size();
    double var_r = 0.0;
    for (double r : logReturns) var_r += (r - mean_r) * (r - mean_r);
    var_r /= (logReturns.size() > 1 ? logReturns.size() - 1 : 1);
    double sigma_r = sqrt(var_r);

    vector<double> truncatedReturns;
    for (double r : logReturns) {
        if (fabs(r - mean_r) <= 3 * sigma_r) {
            truncatedReturns.push_back(r);
        }
    }

    if (truncatedReturns.empty()) {
        cerr << "No valid truncated returns calculated!" << endl;
        return 1;
    }

    int window = min(30, (int)truncatedReturns.size());
    double rollingMean = accumulate(truncatedReturns.end() - window, truncatedReturns.end(), 0.0) / window;

    double sigma = sigma_r;
    double mu = rollingMean + 0.5 * sigma * sigma;

    int days = 30;
    int simulations = 20000;

    double lastPrice = prices[prices.size() - days - 1];
    double actualFuturePrice = prices.back();

    double* d_finalPrices;
    CUDA_CHECK(cudaMalloc(&d_finalPrices, simulations * sizeof(double)));

    int threadsPerBlock = 256;
    int blocks = (simulations + threadsPerBlock - 1) / threadsPerBlock;

    random_device rd;
    unsigned long long seed = rd();
    monteCarloKernel<<<blocks, threadsPerBlock>>>(d_finalPrices, lastPrice, mu, sigma, days, simulations, seed);
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<double> finalPrices(simulations);
    CUDA_CHECK(cudaMemcpy(finalPrices.data(), d_finalPrices, simulations * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_finalPrices));


    double avg = accumulate(finalPrices.begin(), finalPrices.end(), 0.0) / finalPrices.size();
    sort(finalPrices.begin(), finalPrices.end());
    double median = finalPrices[finalPrices.size() / 2];
    double p5 = finalPrices[finalPrices.size() * 0.05];
    double p95 = finalPrices[finalPrices.size() * 0.95];
    double minP = *min_element(finalPrices.begin(), finalPrices.end());
    double maxP = *max_element(finalPrices.begin(), finalPrices.end());

    double mae = fabs(median - actualFuturePrice);
    double mape = fabs((median - actualFuturePrice) / actualFuturePrice) * 100.0;
    bool covered = (actualFuturePrice >= p5 && actualFuturePrice <= p95);

    cout << fixed << setprecision(2);
    cout << "=== GBM + Monte Carlo Prediction - CUDA (30 days) ===" << endl;
    cout << "Start Price (30 days prior): " << lastPrice << endl;
    cout << "Estimated Drift (mu): " << mu << endl;
    cout << "Estimated Volatility (sigma): " << sigma << endl;
    cout << "Predicted Average Price: " << avg << endl;
    cout << "Median Predicted Price: " << median << endl;
    cout << "5th Percentile: " << p5 << " | 95th Percentile: " << p95 << endl;
    cout << "Min Predicted Price: " << minP << " | Max Predicted Price: " << maxP << endl;
    cout << "-----------------------------------" << endl;
    cout << "Actual Price after " << days << " days: " << actualFuturePrice << endl;
    cout << "MAE (Median vs Actual): " << mae << endl;
    cout << "MAPE (Median vs Actual): " << mape << "%" << endl;
    cout << "Coverage (Actual price inside 5%-95% interval): " << (covered ? "YES" : "NO") << endl;

    clock_t end_time = clock();
    cout << "Execution Time: " << double(end_time - start_time) / CLOCKS_PER_SEC << " seconds" << endl;

    return 0;
}