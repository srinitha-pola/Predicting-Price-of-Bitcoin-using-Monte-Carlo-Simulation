#include <bits/stdc++.h>
#include <mpi.h>
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

    // Skip header
    for (int i = 0; i < 3 && getline(file, line); ++i) {}

    while (getline(file, line)) {
        vector<string> tokens = split(line, ',');
        if (tokens.size() < 2) continue;
        try { prices.push_back(stod(tokens[1])); } 
        catch (...) { continue; }
    }
    return prices;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    string filename = "aapl_stock_data.csv";
    vector<double> fullPrices;

    if (rank == 0) {
        fullPrices = readCSV(filename);
        if (fullPrices.size() < 60) {
            cerr << "Not enough price data!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // --- Scatter prices across processes ---
    int chunkSize;
    if (rank == 0) {
        chunkSize = fullPrices.size() / size;
    }
    MPI_Bcast(&chunkSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<double> localPrices(chunkSize);
    MPI_Scatter(fullPrices.data(), chunkSize, MPI_DOUBLE,
                localPrices.data(), chunkSize, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // --- Each process computes log returns ---
    vector<double> logReturns;
    for (size_t i = 1; i < localPrices.size(); i++) {
        if (localPrices[i - 1] != 0) {
            double r = log(localPrices[i] / localPrices[i - 1]);
            logReturns.push_back(r);
        }
    }

    // --- Gather log returns at root ---
    int localCount = logReturns.size();
    vector<int> counts(size);
    MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> displs(size, 0);
    int totalCount = 0;
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            displs[i] = totalCount;
            totalCount += counts[i];
        }
    }

    vector<double> allLogReturns(totalCount);
    MPI_Gatherv(logReturns.data(), localCount, MPI_DOUBLE,
                allLogReturns.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // --- Only rank 0 continues with Monte Carlo ---
    if (rank == 0) {
        if (allLogReturns.empty()) {
            cerr << "No valid log returns calculated!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double mean_r = accumulate(allLogReturns.begin(), allLogReturns.end(), 0.0) / allLogReturns.size();
        double var_r = 0.0;
        for (double r : allLogReturns) var_r += (r - mean_r) * (r - mean_r);
        var_r /= (allLogReturns.size() - 1);
        double sigma_r = sqrt(var_r);

        // Outlier removal
        vector<double> truncatedReturns;
        for (double r : allLogReturns) {
            if (fabs(r - mean_r) <= 3 * sigma_r) truncatedReturns.push_back(r);
        }

        int window = min(30, (int)truncatedReturns.size());
        double rollingMean = accumulate(truncatedReturns.end() - window, truncatedReturns.end(), 0.0) / window;

        double sigma = sigma_r;
        double mu = rollingMean + 0.5 * sigma * sigma;

        int days = 30;
        int simulations = 20000;

        double lastPrice = fullPrices[fullPrices.size() - days - 1];
        double actualFuturePrice = fullPrices.back();

        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> norm(0.0, 1.0);

        vector<double> finalPrices(simulations);

        // --- OpenMP parallel Monte Carlo ---
        #pragma omp parallel for
        for (int i = 0; i < simulations; i++) {
            double price = lastPrice;
            for (int d = 0; d < days; d++) {
                double z = norm(gen);
                double dt = 1.0;
                price *= exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * z);
            }
            finalPrices[i] = price;
        }

        // --- Compute statistics ---
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
        cout << "=== GBM + Monte Carlo Prediction using OpenMP and MPI(30 days) ===" << endl;
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

        double end_time = MPI_Wtime();
        cout << "Execution Time: " << (end_time - start_time) << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}