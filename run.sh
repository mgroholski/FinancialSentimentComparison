#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -t 1-12:00:00   # time in d-hh:mm:ss
#SBATCH -p public      # partition
#SBATCH -q public       # QOS
#SBATCH -o /home/mgrohols/logs/575logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e /home/mgrohols/logs/575logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment
#SBATCH --mem=100G
#SBATCH -G 1

module load mamba
source activate tfenv

cd /scratch/mgrohols/575/financial-sentiment

python3 main.py -d ./data_utils/price_news_integrate -p lstm -r 3
