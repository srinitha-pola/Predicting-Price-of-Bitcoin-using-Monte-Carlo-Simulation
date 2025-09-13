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

    if (!file.is_open()) {
        cout << "Error: Could not open file " << filename << endl;
        return prices;
    }

    for (int i = 0; i < 3 && getline(file, line); ++i) {}

    while (getline(file, line)) {
        vector<string> tokens = split(line, ',');
        if (tokens.size() < 2) continue;
        try {
            double close_price = stod(tokens[1]);
            prices.push_back(close_price);
        } catch (const std::invalid_argument& e) {
            continue;
        } catch (const std::out_of_range& e) {
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
        if (prices[i - 1] != 0) {
            logReturns.push_back(log(prices[i] / prices[i - 1]));
        }
    }

    if (logReturns.empty()) {
        cout << "No valid log returns calculated!" << endl;
        return 1;
    }

    double mean = accumulate(logReturns.begin(), logReturns.end(), 0.0) / logReturns.size();
    double variance = 0.0;
    for (double r : logReturns) variance += (r - mean) * (r - mean);
    variance /= (logReturns.size() - 1);
    double sigma = sqrt(variance);

    int simulations = 1000;
    int days = 30;
    double lastPrice = prices.back();
    vector<double> finalPrices;

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> norm(0.0, 1.0);

    for (int i = 0; i < simulations; i++) {
        double price = lastPrice;
        for (int d = 0; d < days; d++) {
            double z = norm(gen);
            price *= exp(mean + sigma * z);
        }
        finalPrices.push_back(price);
    }

    double avg = accumulate(finalPrices.begin(), finalPrices.end(), 0.0) / finalPrices.size();
    sort(finalPrices.begin(), finalPrices.end());

    double median = finalPrices[finalPrices.size() / 2];
    double p5 = finalPrices[finalPrices.size() * 0.05];
    double p95 = finalPrices[finalPrices.size() * 0.95];
    
    double minP = *min_element(finalPrices.begin(), finalPrices.end());
    double maxP = *max_element(finalPrices.begin(), finalPrices.end());

    cout << fixed << setprecision(2);
    cout << "Estimated Mean Log Return: " << mean << endl;
    cout << "Estimated Volatility (sigma): " << sigma << endl;
    cout << "Monte Carlo Bitcoin Price Prediction (" << days << " days)" << endl;
    cout << "Current Price: " << lastPrice << endl;
    cout << "Predicted Average Price: " << avg << endl;
    cout << "Median: " << median << endl;
    cout << "5th Percentile: " << p5 << " | 95th Percentile: " << p95 << endl;
    cout << "Min: " << minP << " | Max: " << maxP << endl;

    return 0;
}