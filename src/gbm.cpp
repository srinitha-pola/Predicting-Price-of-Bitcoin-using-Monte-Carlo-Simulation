#include <bits/stdc++.h>
#include <chrono>
using namespace std;

vector<string> split(const string& s, char delimiter) {
    vector<string> tokens;
    string token;
    stringstream ss(s);
    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

vector<double> readCSV(const string& filename) {
    ifstream file(filename);
    vector<double> prices;
    string line;

    for (int i = 0; i < 3 && getline(file, line); ++i) {} // skip headers

    while (getline(file, line)) {
        vector<string> tokens = split(line, ',');
        if (tokens.size() < 2) continue;
        try {
            double close_price = stod(tokens[1]);
            prices.push_back(close_price);
        } catch (...) {
            continue;
        }
    }
    return prices;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    string filename = "aapl_stock_data.csv";
    vector<double> prices = readCSV(filename);

    if (prices.size() < 60) { 
        cout << "Not enough price data!" << endl;
        return 1;
    }

    int days = 30;
    vector<double> train(prices.begin(), prices.end() - days);
    vector<double> test(prices.end() - days, prices.end());

    vector<double> logReturns;
    for (size_t i = 1; i < train.size(); i++) {
        logReturns.push_back(log(train[i] / train[i - 1]));
    }

    double mean = accumulate(logReturns.begin(), logReturns.end(), 0.0) / logReturns.size();
    double variance = 0.0;
    for (double r : logReturns) variance += (r - mean) * (r - mean);
    variance /= (logReturns.size() - 1);
    double sigma = sqrt(variance);
    double mu = mean + 0.5 * sigma * sigma;

    double lastPrice = train.back();
    vector<double> gbmPrices;
    gbmPrices.push_back(lastPrice);

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> norm(0.0, 1.0);

    for (int d = 1; d <= days; d++) {
        double z = norm(gen);
        double nextPrice = gbmPrices.back() * exp((mu - 0.5 * sigma * sigma) + sigma * z);
        gbmPrices.push_back(nextPrice);
    }

    double mae = 0.0, mse = 0.0, mape = 0.0;
    for (int i = 0; i < days; i++) {
        double actual = test[i];
        double predicted = gbmPrices[i + 1];
        mae += fabs(predicted - actual);
        mse += (predicted - actual) * (predicted - actual);
        mape += fabs((predicted - actual) / actual);
    }
    mae /= days;
    mse /= days;
    mape = (mape / days) * 100.0;

    cout << "Estimated Drift (mu): " << mu << endl;
    cout << "Estimated Volatility (sigma): " << sigma << endl;

    cout << "\nAccuracy Metrics:" << endl;
    cout << "MAE: " << mae << endl;
    cout << "RMSE: " << sqrt(mse) << endl;
    cout << "MAPE: " << mape << " %" << endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count() << " ms" << endl;

    return 0;
}