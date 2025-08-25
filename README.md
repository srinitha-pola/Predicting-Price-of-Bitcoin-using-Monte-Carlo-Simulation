# Bitcoin Price Prediction using Monte Carlo Simulation and HPC (MPI) 🚀

## 📌 Overview
This project predicts possible future Bitcoin prices using **Monte Carlo simulation**, which generates many random paths of possible outcomes based on statistical models. Since this requires high computational power, we use **High Performance Computing (HPC)** with **MPI** to parallelize the simulations across multiple nodes.  

The outcome is a probability distribution of Bitcoin’s future price, with future scope to integrate **market sentiment analysis** for improved predictions.

---

## 🎯 Problem Statement
Monte Carlo simulations require thousands to millions of paths to get accurate results. Running these simulations sequentially is computationally expensive.  

HPC with MPI allows splitting the workload across multiple processors/nodes to achieve faster results. Our aim is to:
- Implement a sequential Monte Carlo simulator.
- Implement a parallelized version using MPI.
- Compare performance and scalability between the two.

---

## ⚙️ Methodology / Approach

### Phase 1: Sequential CPU Implementation
- Implement Bitcoin price simulation using **Geometric Brownian Motion (GBM)**.
- Validate correctness with small number of simulations.

### Phase 2: MPI Parallelization
- Split simulation tasks across multiple processes.
- Use **mpi4py** to implement distributed execution.
- Aggregate results (mean, variance, percentiles).

### Phase 3: Performance Analysis
- Measure execution time with varying number of simulations.
- Plot **speedup vs number of processes**.
- Analyze scalability and efficiency.

---

## 📐 Mathematical Model
We use **Geometric Brownian Motion (GBM)** as the stochastic process for Bitcoin prices:

\[
S_{t+1} = S_t \cdot e^{\left( \mu - \tfrac{1}{2}\sigma^2 \right)\Delta t + \sigma \sqrt{\Delta t}\, Z_t}, 
\quad Z_t \sim \mathcal{N}(0,1)
\]

Where:
- \( \mu \) = drift (average return)
- \( \sigma \) = volatility
- \( Z_t \) = random variable from standard normal distribution

---

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Libraries**: numpy, pandas, matplotlib, mpi4py
- **Tools**: GitHub, MPI (OpenMPI / MPICH)

---

## 📂 Repository Structure
Predicting-Price-of-Bitcoin-using-Monte-Carlo-Simulation/
├── README.md                # Project overview (already created)
├── requirements.txt         # Python dependencies
├── src/                     # Source code
├── docs/                    # Documentation (reports, diagrams, PDFs)
│   └── project_report.pdf

---

## ▶️ How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/srinitha-pola/Predicting-Price-of-Bitcoin-using-Monte-Carlo-Simulation.git
cd Predicting-Price-of-Bitcoin-using-Monte-Carlo-Simulation
```
### 2. Set Up Environment
---------------------
It is recommended to use a virtual environment.

Linux / macOS:
    ``` bash
    python3 -m venv venv
    source venv/bin/activate
    ```

Windows (PowerShell):
  ```bash  python -m venv venv
    .\venv\Scripts\activate
```

### 3. Install Dependencies
-----------------------
```bash pip install -r requirements.txt
```
If mpi4py fails, install MPI first:
- Ubuntu/Debian:```bash sudo apt-get install libopenmpi-dev openmpi-bin```
- Windows: ```bash Install MS-MPI (https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)```

Then reinstall:
   ```bash pip install mpi4py```

4. Run Sequential Monte Carlo Simulation
----------------------------------------
```bash python src/montecarlo.py --nsims 10000 --horizon 30 --s0 30000 --mu 0.0005 --sigma 0.04```

Example output:
    Mean final price: 30125.63
    Std final price: 1205.45

5. Run Parallel MPI Version
---------------------------
Use mpiexec to distribute work across processes.

```bash mpiexec -n 4 python src/mpi_worker.py --nsims 100000 --horizon 30 --s0 30000 --mu 0.0005 --sigma 0.04```

- -n 4 → number of processes
- --nsims → total simulations (split across processes)
- --horizon → days to predict
- --s0 → initial price
- --mu → drift
- --sigma → volatility

Example output:
    Global mean final price: 30120.47

6. Running on HPC Cluster (Optional)
------------------------------------
If using SLURM or another scheduler:
```bash srun -n 16 python src/mpi_worker.py --nsims 1000000 --horizon 30 --s0 30000 --mu 0.0005 --sigma 0.04```

7. Analyze Results
------------------
Store outputs (CSV/plots) in results/ and visualize them with:

```bash python src/analyze.py --input results/simulations.csv --plot results/histogram.png```

This produces histograms and performance plots.


