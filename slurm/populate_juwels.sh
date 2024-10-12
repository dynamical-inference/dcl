#!/bin/bash
#SBATCH --account=hai_mechanistic
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=dcl
#SBATCH --output=slurm_output/worker_%A_%a_%j.out
#SBATCH --signal=TERM@300

#source $MINICONDA/etc/profile.d/conda.sh
#conda activate dynamics
#echo "activating conda env: dynamics"

#conda env list
#which python
#python -c "import torch; print('torch', torch.__version__)"
#python -c "import torch; print('torch:is_cuda_available', torch.cuda.is_available())"
#python -c "import numpy; print('numpy', numpy.__version__)"
##python -c "import skleanr; print('numpy', numpy.__version__)"


GIT_PATH="/p/project1/hai_mechanistic/thekings/git/"
if ! echo "$PATH" | grep -q "$GIT_PATH"; then
    export PATH="$GIT_PATH:$PATH"
fi
export GIT_PYTHON_GIT_EXECUTABLE="$GIT_PATH/git"
git status
python -c "import git; print(git.Repo(search_parent_directories=True))"

# Create slurm_output directory if it doesn't exist
mkdir -p slurm_output

nvidia-smi

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

srun --output slurm_output/log_%A_%a_%t.out \
        python dcl/datajoint/populate.py \
        --order random \
        --suppress-errors \
        --sweep "$SWEEP" \
        $EVAL_ONLY_ARG \
        $LIMIT_ARG
