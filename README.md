2. Set Up Environment
---------------------
It is recommended to use a virtual environment.

Linux / macOS:
    python3 -m venv venv
    source venv/bin/activate

Windows (PowerShell):
    python -m venv venv
    .\venv\Scripts\activate

3. Install Dependencies
-----------------------
pip install -r requirements.txt

If mpi4py fails, install MPI first:
- Ubuntu/Debian: sudo apt-get install libopenmpi-dev openmpi-bin
- macOS: brew install open-mpi
- Windows: Install MS-MPI (https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)

Then reinstall:
    pip install mpi4py

4. Run Sequential Monte Carlo Simulation
----------------------------------------
python src/montecarlo.py --nsims 10000 --horizon 30 --s0 30000 --mu 0.0005 --sigma 0.04

Example output:
    Mean final price: 30125.63
    Std final price: 1205.45

5. Run Parallel MPI Version
---------------------------
Use mpiexec to distribute work across processes.

mpiexec -n 4 python src/mpi_worker.py --nsims 100000 --horizon 30 --s0 30000 --mu 0.0005 --sigma 0.04

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
srun -n 16 python src/mpi_worker.py --nsims 1000000 --horizon 30 --s0 30000 --mu 0.0005 --sigma 0.04

7. Analyze Results
------------------
Store outputs (CSV/plots) in results/ and visualize them with:

python src/analyze.py --input results/simulations.csv --plot results/histogram.png

This produces histograms and performance plots.
