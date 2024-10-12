# Running Experiments for Reproducing Table 1

This repository contains code and configuration to reproduce the experiments reported in **Table 1** of our paper. Follow the steps below to set up your environment and launch the experiments.

### 📥 Step 1: Dataset Download

Download the datasets as described in the main README. Save the files in a directory and update the `basepath` variable in the `reproduce_table1.py` file.

### 🛠️ Step 2: Setup DataJoint Connection

To get started, make sure you can connect to a **DataJoint database**  and set up the required environment variables. If you have difficulties setting up the database, please contact us.

```bash
DATAJOINT_DB_HOST="xx.xx.xx.xx:xxxx"
DATAJOINT_DB_USER="root"
DATAJOINT_DB_PASSWORD=""
DATAJOINT_SCHEMA_NAME="table1_reproduce"
```

These variables will be automatically loaded if you're using a .env file.  Otherwise, set them manually in your shell environment.

### 🧪 Step 3: Define the Experiment Sweep

All experiments are configured through sweep files located in the sweeps/ directory. For example:

```python
python sweeps/reproduce_table1.py
```

This script will populate the database with experiment configurations associated with the sweep name "Table 1". Feel free to create additional sweep scripts for other experiments by following the same structure.

### 🚀 Step 4: Launch Jobs Using SLURM

Once the sweep is created, allocate the jobs using sbatch. Here's an example command:

```bash
sbatch --array=1-12 --ntasks=4 slurm/populate.sh --sweep='Table 1'
```

You can modify this command to match your specific cluster configuration and experiment needs.


## Reference Results

| Dynamics Model                           | Model              | R2_Backward_bias    | R2_Forward_bias     |
|------------------------------------------|:-------------------|:-------------------|:-------------------|
| Identity                                 | Identity          | 99.6 ± 0.16 (n=9)  | 99.7 ± 0.10 (n=9)  |
| Identity                                 | LDS               | 99.5 ± 0.09 (n=9)  | 99.6 ± 0.09 (n=9)  |
| LDS                                      | Identity          | 81.2 ± 18.33 (n=9) | 73.8 ± 37.01 (n=9) |
| LDS                                      | LDS               | 98.8 ± 0.47 (n=9)  | 98.3 ± 0.79 (n=9)  |
| LDS (low dt)                            | Identity          | 88.2 ± 6.02 (n=9)  | 90.4 ± 11.46 (n=9) |
| LDS (low dt)                            | LDS               | 98.3 ± 0.68 (n=9)  | 98.2 ± 0.61 (n=9)  |
| SLDS                                     | Identity          | 77.2 ± 6.38 (n=9)  | 82.0 ± 7.23 (n=9)  |
| SLDS                                     | SLDS (5)          | 99.5 ± 0.04 (n=9)  | 99.6 ± 0.06 (n=9)  |
| Lorenz                                   | Identity          | 45.5 ± 8.94 (n=15) | 21.6 ± 6.66 (n=15) |
| Lorenz                                   | LDS               | 82.4 ± 18.41 (n=15)| 85.0 ± 11.94 (n=15)|
| Lorenz                                   | SLDS (200)        | 90.4 ± 3.51 (n=15) | 88.8 ± 9.41 (n=15) |
| Lorenz (small dt, large var)            | Identity          | 99.9 ± 0.12 (n=15) | 99.8 ± 0.18 (n=15) |
| Lorenz (small dt, large var)            | LDS               | 98.5 ± 2.14 (n=15) | 95.7 ± 6.16 (n=15) |
| Lorenz (small dt, large var)            | SLDS (200)        | 93.9 ± 4.87 (n=15) | 89.7 ± 9.72 (n=15) |



Note: Some results, especially on the Lorenz system, require exact matches in the codebase and the environment setup.
If you are interested in reproducing these, we can share compute environment and the original research codebase on request (we might update this repository in the future to contain these environments based on demand)
