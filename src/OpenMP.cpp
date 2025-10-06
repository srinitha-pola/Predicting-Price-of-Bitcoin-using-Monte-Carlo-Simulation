#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

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
        try { prices.push_back(stod(tokens[1])); } 
        catch (...) { continue; }
    }
    return prices;
}

int main() {
    double start_time = omp_get_wtime();

    string filename = "aapl_stock_data.csv";
    vector<double> prices = readCSV(filename);

    if (prices.size() < 60) {
        cerr << "Not enough price data!" << endl;
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
    var_r /= (logReturns.size() - 1);
    double sigma_r = sqrt(var_r);

    vector<double> truncatedReturns;
    for (double r : logReturns) {
        if (fabs(r - mean_r) <= 3 * sigma_r) truncatedReturns.push_back(r);
    }

    int window = min(30, (int)truncatedReturns.size());
    double rollingMean = accumulate(truncatedReturns.end() - window, truncatedReturns.end(), 0.0) / window;

    double sigma = sigma_r;
    double mu = rollingMean + 0.5 * sigma * sigma;

    int days = 30;
    int simulations = 20000;

    double lastPrice = prices[prices.size() - days - 1];
    double actualFuturePrice = prices.back();

    vector<double> finalPrices(simulations);
    #pragma omp parallel
    {
        random_device rd;
        mt19937 gen(rd() + omp_get_thread_num());
        normal_distribution<> norm(0.0, 1.0);

        #pragma omp for
        for (int i = 0; i < simulations; i++) {
            double price = lastPrice;
            for (int d = 0; d < days; d++) {
                double z = norm(gen);
                double dt = 1.0;
                price *= exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * z);
            }
            finalPrices[i] = price;
        }
    }

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
    cout << "=== GBM + Monte Carlo Prediction with OpenMP===" << endl;
    cout << "Start Price: " << lastPrice << endl;
    cout << "Estimated Drift (mu): " << mu << endl;
    cout << "Estimated Volatility (sigma): " << sigma << endl;
    cout << "Predicted Average Price: " << avg << endl;
    cout << "Median: " << median << endl;
    cout << "5th Percentile: " << p5 << " | 95th Percentile: " << p95 << endl;
    cout << "Min: " << minP << " | Max: " << maxP << endl;
    cout << "-----------------------------------" << endl;
    cout << "Actual Price after " << days << " days: " << actualFuturePrice << endl;
    cout << "MAE: " << mae << endl;
    cout << "MAPE: " << mape << "%" << endl;
    cout << "Coverage (inside 5%-95% interval): " << (covered ? "YES" : "NO") << endl;

    double end_time = omp_get_wtime();
    cout << "Execution Time: " << (end_time - start_time) << " seconds" << endl;

    return 0;
}
