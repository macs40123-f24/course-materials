#!/bin/bash

#SBATCH --job-name=ssd-spark-example
#SBATCH --output=ssd-spark.out
#SBATCH --error=ssd-spark.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=40G
#SBATCH --partition=ssd
#SBATCH --account=ssd

module load python/anaconda-2022.05 spark/3.3.2

export PYSPARK_DRIVER_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3
export PYSPARK_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3

spark-submit --total-executor-cores 8 --executor-memory 5G minhashlsh.py

# to filter out logging messages from output, run the following:
# cat ssd-spark.out | grep -vE "INFO|WARN"
