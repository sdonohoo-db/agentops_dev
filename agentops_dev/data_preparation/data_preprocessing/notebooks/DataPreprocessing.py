# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# DBTITLE 1,Data Preprocessing Pipeline - Overview
################################################################################### 
# Data Preprocessing Pipeline
#
# This notebook shows an example of a Data Preprocessing pipeline using Unity Catalog.
# It is configured and can be executed as the tasks in the PreprocessRawData workflow defined under
# ``agentops_dev/resources/data-preprocessing-workflow-resource.yml``
#
# Parameters:
# * uc_catalog (required)                     - Name of the Unity Catalog
# * schema (required)                         - Name of the schema inside Unity Catalog
# * raw_data_table (required)                 - Name of the raw data table inside UC database
# * preprocessed_data_table (required)        - Name of the preprocessed data table inside UC database
# * hf_tokenizer_model (optional)             - Name of the HuggingFace tokenizer model name
# * max_chunk_size (optional)                 - Maximum chunk size
# * min_chunk_size (optional)                 - Minimum chunk size
# * chunk_overlap (optional)                  - Overlap between chunks
#
# Widgets:
# * Unity Catalog: Text widget to input the name of the Unity Catalog
# * schema: Text widget to input the name of the database inside the Unity Catalog
# * Raw data table: Text widget to input the name of the raw data table inside the database of the Unity Catalog
# * Preprocessed data table: Text widget to input the name of the preprocessed data table inside the database of the Unity Catalog
# * HuggingFace tokenizer model: Text widget to input the name of the hugging face tokenizer model to import
# * Maximum chunk size: Maximum characters chunks will be split into
# * Minimum chunk size: minimum characters chunks will be split into
# * Chunk overlap: Overlap between chunks
#
# Usage:
# 1. Set the appropriate values for the widgets.
# 2. Run the pipeline to chunk the raw data and store in Unity Catalog.
#
##################################################################################

# COMMAND ----------

# DBTITLE 1,Install Prerequisites
# Install prerequisite packages
%pip install -r ../../data_prep_requirements.txt

# COMMAND ----------

# DBTITLE 1,Widget Creation
# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# A Unity Catalog containing the input data
dbutils.widgets.text(
    "uc_catalog",
    "",
    label="Unity Catalog",
)
# Name of schema
dbutils.widgets.text(
    "schema",
    "",
    label="Schema",
)
# Name of input table
dbutils.widgets.text(
    "raw_data_table",
    "raw_documentation",
    label="Raw data table",
)
# Name of output table
dbutils.widgets.text(
    "preprocessed_data_table",
    "databricks_documentation",
    label="Preprocessed data table",
)

# COMMAND ----------

# DBTITLE 1,Define input and output variables
uc_catalog = dbutils.widgets.get("uc_catalog")
schema = dbutils.widgets.get("schema")
raw_data_table = dbutils.widgets.get("raw_data_table")
preprocessed_data_table = dbutils.widgets.get("preprocessed_data_table")

assert uc_catalog != "", "uc_catalog notebook parameter must be specified"
assert schema != "", "schema notebook parameter must be specified"
assert raw_data_table != "", "raw_data_table notebook parameter must be specified"
assert preprocessed_data_table != "", "preprocessed_data_table notebook parameter must be specified"

# COMMAND ----------

# DBTITLE 1,Set up path to import utility functions
import sys
import os

# Get notebook's directory using dbutils
notebook_path = '/Workspace/' + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook()
    .getContext().notebookPath().get()
)
# Navigate up from notebooks/ to component level
utils_dir = os.path.dirname(notebook_path)
sys.path.insert(0, utils_dir)

# COMMAND ----------

# DBTITLE 1,Import Chunking Configuration
# Import chunking configuration constants
from utils.config import (
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    CHUNK_OVERLAP,
    HF_TOKENIZER_MODEL
)

min_chunk_size = MIN_CHUNK_SIZE
max_chunk_size = MAX_CHUNK_SIZE
chunk_overlap = CHUNK_OVERLAP
hf_tokenizer_model = HF_TOKENIZER_MODEL

# COMMAND ----------

# DBTITLE 1,Initialize tokenizer
# Download tokenizer model to UC volume
from transformers import AutoTokenizer

volume_folder =  f"/Volumes/{uc_catalog}/{schema}/volume_databricks_documentation"

spark.sql(f"CREATE VOLUME IF NOT EXISTS {uc_catalog}.{schema}.volume_databricks_documentation")

# Initialize tokenizer once
tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_model, cache_dir=f'{volume_folder}/hg_cache')


# COMMAND ----------

# DBTITLE 1, Use the catalog and database specified in the notebook parameters
spark.sql(f"""USE `{uc_catalog}`.`{schema}`""")   

# COMMAND ----------

# DBTITLE 1, Create output preprocessed data table
if not spark.catalog.tableExists(f"{preprocessed_data_table}") or spark.table(f"{preprocessed_data_table}").isEmpty():
  spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {preprocessed_data_table} (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    url STRING,
    content STRING
  )
  TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
  """)


# COMMAND ----------

# DBTITLE 1,Create a user-defined function (UDF) to chunk all our documents with spark.
from functools import partial
import pandas as pd
from pyspark.sql.functions import pandas_udf
from utils.create_chunk import split_html_on_p

@pandas_udf("array<string>")
def parse_and_split(
    docs: pd.Series
) -> pd.Series:
    """Parse and split html content into chunks.

    :param docs: Input documents
    :return: List of chunked text for each input document
    """
    
    return docs.apply(lambda html: split_html_on_p(
        html,
        tokenizer=tokenizer,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size
    ))

# COMMAND ----------

# DBTITLE 1,Perform data preprocessing.
from pyspark.sql import functions as F

(spark.table(raw_data_table)
      .filter('text is not null')
      .withColumn('content', F.explode(parse_and_split('text')))
      .drop("text")
      .write.mode('overwrite').saveAsTable(preprocessed_data_table))

# COMMAND ----------

# DBTITLE 1,Exit notebook
dbutils.notebook.exit(0)
