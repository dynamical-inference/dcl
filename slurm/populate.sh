#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=dcl
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:full:1
#SBATCH --output=slurm_output/worker_%A_%a_%j.out
#SBATCH --signal=TERM@300

### Setup Environment

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME
echo "activating conda env: $CONDA_ENV_NAME"

conda env list
which python
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch; print('torch:is_cuda_available', torch.cuda.is_available())"
python -c "import numpy; print('numpy', numpy.__version__)"

nvidia-smi

nvidia-smi -L
output=$(nvidia-smi -L)
# Use grep to find lines containing 'GPU', then use awk to trim and print the UUID part
uuids=$(echo "$output" | grep "GPU" | awk -F'UUID: ' '{gsub(/\).*/, "", $2); print $2}')

echo "uuids=$uuids"
### Parse Arguments



# Default values for parameters
EVAL_ONLY=False
LIMIT=False

# Function to handle the SIGTERM signal
terminate() {
    echo "Caught SIGTERM signal! Terminating child processes..."
    # Send SIGTERM to all background processes
    kill $(jobs -p)
    wait
    echo "All background processes terminated."
}



# Parse named arguments
for i in "$@"
do
case $i in
    --sweep=*)
    SWEEP="${i#*=}"
    shift # past argument=value
    ;;
    --eval_only)
    EVAL_ONLY=True
    shift # past argument
    ;;
    --limit)
    LIMIT=True
    shift # past argument
    ;;
    *)
          # unknown option
    ;;
esac
done

# Check if SWEEP variable is set
if [ -z "$SWEEP" ]; then
  echo "Need to specify --sweep."
  exit 1
fi

if [ "$EVAL_ONLY" = True ]; then
    EVAL_ONLY_ARG="--eval_only"
else
    EVAL_ONLY_ARG=""
fi

if [ "$LIMIT" = True ]; then
    LIMIT_ARG="--limit 1"
else
    LIMIT_ARG=""
fi


# Get GPU UUIDs and store them in an array
nvidia-smi -L
output=$(nvidia-smi -L)
# Convert UUIDs into an array
readarray -t gpu_uuids < <(echo "$output" | grep "GPU" | awk -F'UUID: ' '{gsub(/\).*/, "", $2); print $2}')
num_gpus=${#gpu_uuids[@]}

echo "Available GPUs: ${num_gpus}"
echo "UUIDs: ${gpu_uuids[*]}"

# Create a comma-separated string of UUIDs for passing to srun
gpu_uuid_string=$(IFS=,; echo "${gpu_uuids[*]}")

# Echo the command that will be run
echo "Will run command:"
echo "python dcl/datajoint/populate.py --order random --suppress-errors --sweep $SWEEP $EVAL_ONLY_ARG $LIMIT_ARG"


# Launch multiple tasks and set CUDA_VISIBLE_DEVICES using UUID
srun --output slurm_output/log_%A_%a_%t.out \
    bash -c 'IFS="," read -ra UUIDS <<< "'"$gpu_uuid_string"'"; \
    gpu_index=$((SLURM_LOCALID % ${#UUIDS[@]})); \
    export CUDA_VISIBLE_DEVICES="${UUIDS[$gpu_index]}"; \
    python dcl/datajoint/populate.py \
        --order random \
        --suppress-errors \
        --sweep "'"$SWEEP"'" '$EVAL_ONLY_ARG' '$LIMIT_ARG''

echo "Done"
