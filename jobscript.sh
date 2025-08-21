#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J 2ES
### -- ask for number of cores --
#BSUB -n 12
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- OPTIMIZED: reduce memory per core from 4GB to 1GB --
#BSUB -R "rusage[mem=256MB]"
### -- OPTIMIZED: reduce memory limit from 5GB to 2GB per core --
#BSUB -M 256MB
### -- walltime  --
#BSUB -W 5:00
### -- set the email address --
#BSUB -u jzhao@space.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file --
#BSUB -o /work2/lbuc/jzhao/2ES_target_selection/logfiles/2ES_target_selection.out

# Print job information
echo "=== Job Information ==="
echo "Job ID: $LSB_JOBID"
echo "Job started at: $(date)"
echo "Running on host: $HOSTNAME"
echo "Available cores: $LSB_DJOB_NUMPROC"

# Change to working directory
cd /work2/lbuc/jzhao/2ES_target_selection || {
    echo "ERROR: Cannot change to working directory"
    exit 1
}

# Create logfiles directory
mkdir -p logfiles

echo "Working directory: $(pwd)"

# Activate Conda environment with error checking
echo "=== Activating Conda Environment ==="
source /zhome/9d/b/207249/anaconda3/etc/profile.d/conda.sh || {
    echo "ERROR: Failed to source conda"
    exit 1
}

conda activate 2ES || {
    echo "ERROR: Failed to activate 2ES environment"
    echo "Available environments:"
    conda env list
    exit 1
}

echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"


# Run the analysis
echo "=== Starting 2ES Analysis ==="
cd src && python 2ES.py

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo "=== 2ES Analysis Completed Successfully ==="
else
    echo "=== 2ES Analysis Failed ==="
    exit 1
fi

echo "Job completed at: $(date)"

# Print resource usage summary
echo "=== Resource Usage Summary ==="
echo "Job ID: $LSB_JOBID"
echo "Cores used: $LSB_DJOB_NUMPROC"
echo "Host: $LSB_HOSTS"
echo "Max memory used: $(bjobs -l $LSB_JOBID 2>/dev/null | grep -o 'MAX MEM: [0-9]*' || echo 'N/A')"