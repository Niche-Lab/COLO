#!/bin/bash
#SBATCH --job-name=InTeRaCtIvE
#SBATCH -t 143:59:59
#SBATCH --partition=l40s_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=niche_squad
#SBATCH --output=InTeRaCtIvE.out

while true
do
  echo "This loop will run forever..."
  sleep 10
done
