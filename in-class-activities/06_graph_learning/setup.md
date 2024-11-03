## Week 6 Setup Instructions

To run the demo code this week for PyTorch Geometric on Midway 3, you will need to complete the following setup from the login node:

1. Load the default Python (`python/anaconda-2022.05`) and CUDA (`cuda/11.7`) modules:
    ```bash
    module load python cuda
    ```
2. Install PyTorch Geometric and other related packages we will be using on Midway 3 (this will take ~10 minutes to install):
    ```bash
    pip install --user torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
    ```

## Batch Jobs

You should then be able to run the demo code as a batch job via:

```bash
sbatch node2vec.sbatch
```

Once the job is finished running, it should produce output similar to what is provided in `./expected_output/`.

## Interactive Sessions

Or, you can run the same code in an interactive session by requesting the same resources from the login node as in `node2vec.sbatch`:

```bash
sinteractive \
--time=00:30:00 \
--nodes=1 \
--partition=gpu \
--ntasks=1 \
--gres=gpu:1 \
--account=macs40123 \
--mem-per-cpu=30G
```

Once in the interactive session, you can either continue to work from the terminal, or start a Jupyter session (where `$HOST_IP` is the IP address for your compute node on Midway 3):

```bash
module load python cuda
jupyter lab --no-browser --ip=$HOST_IP --port=8888
```

Then, you should be able to follow [the same port forwarding instructions from earlier in the course](https://github.com/macs40123-f24/course-materials/blob/main/in-class-activities/02_pagerank_association/spark_pagerank_interactive.ipynb) to access your Jupyter session through your local browser.

The notebook `./node2vec.ipynb` is provided as an example that you can now run interactively in your Jupyter session.