# Twitch_RS_Project

## Environment Setup

- data can be downloaded from [here](https://cseweb.ucsd.edu//~jmcauley/datasets.html#twitch)
- put data into a folder called data in the home directory
- Need to install spark and hadoop and configure environment variables correctly.

### Libraries

- Use conda for package management, use default env or spark will complain
- PyArrow `conda install pyarrow`
- PySpark `conda install pyspark`
- PySpark.pandas (extension of PySpark) `conda install pandas`

## What order to run

- Run the New_Indexing_Base_PreprocessingNB.ipynb
- Rename csv in data/collab to collab_filter.csv
- Rename csv in data/item to means.csv
- Run ClusterRecV2.ipynb
- Run run_als_explicit.py
