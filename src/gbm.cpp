#include <bits/stdc++.h>
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

    for (int i = 0; i < 3 && getline(file, line); ++i) {} 

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
    string filename = "bitcoin(3).csv";
    vector<double> prices = readCSV(filename);

    if (prices.empty()) {
        cout << "No price data loaded!" << endl;
        return 1;
    }

    vector<double> logReturns;
    for (size_t i = 1; i < prices.size(); i++) {
        logReturns.push_back(log(prices[i] / prices[i - 1]));
    }

    double mean = accumulate(logReturns.begin(), logReturns.end(), 0.0) / logReturns.size();
    double variance = 0.0;
    for (double r : logReturns) variance += (r - mean) * (r - mean);
    variance /= (logReturns.size() - 1);
    double sigma = sqrt(variance);
    double mu = mean + 0.5 * sigma * sigma;

    int days = 30; 
    double lastPrice = prices.back();
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

    cout << "Estimated Drift (mu): " << mu << endl;
    cout << "Estimated Volatility (sigma): " << sigma << endl;
    cout << "\nGBM Forecast for next " << days << " days:" << endl;
    for (int d = 0; d < gbmPrices.size(); d++) {
        cout << "Day " << d << " : " << gbmPrices[d] << endl;
    }

    return 0;
}
