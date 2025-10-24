# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# DBTITLE 1,Agent Evaluation Pipeline - Overview
##################################################################################
# Agent Evaluation
# 
# Notebook that downloads an evaluation dataset and evaluates the model using
# llm-as-a-judge with the Databricks agent framework.
#
# Parameters:
# * uc_catalog (required)           - Name of the Unity Catalog
# * schema (required)               - Name of the schema inside Unity Catalog
# * eval_table (required)           - Name of the table containing the evaluation dataset
# * experiment (required)           - Name of the experiment to register the run under
# * registered_model (required)     - Name of the model registered in mlflow
# * model_alias (required)          - Model alias to deploy
#
# Widgets:
# * Unity Catalog: Text widget to input the name of the Unity Catalog
# * Schema: Text widget to input the name of the database inside the Unity Catalog
# * Evaluation Table: Text widget to input the name of the table containing the evaluation dataset
# * Experiment: Text widget to input the name of the experiment to register the run under
# * Registered model name: Text widget to input the name of the model to register in mlflow
# * Model Alias: Text widget to input the model alias to deploy
#
# Usage:
# 1. Set the appropriate values for the widgets.
# 2. Run to evaluate your agent.
#
##################################################################################

# COMMAND ----------

# DBTITLE 1,Install Prerequisites and Restart Python
# Install prerequisite packages
%pip install -r ../../agent_requirements.txt
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Widget Creation
# List of input args needed to run the notebook as a job.
# Provide them via DB widgets or notebook arguments.

# A Unity Catalog containing the model
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
# Name of evaluation table
dbutils.widgets.text(
    "eval_table",
    "databricks_documentation_eval",
    label="Evaluation dataset",
)
# Name of experiment to register under in mlflow
dbutils.widgets.text(
    "experiment",
    "/agent_function_chatbot",
    label="Experiment name",
)
# Name of model registered in mlflow
dbutils.widgets.text(
    "registered_model",
    "agent_function_chatbot",
    label="Registered model name",
)
# Model alias
dbutils.widgets.text(
    "model_alias",
    "agent_latest",
    label="Model Alias",
)

# COMMAND ----------

# DBTITLE 1,Define Input Variables
uc_catalog = dbutils.widgets.get("uc_catalog")
schema = dbutils.widgets.get("schema")
eval_table = dbutils.widgets.get("eval_table")
experiment = dbutils.widgets.get("experiment")
registered_model = dbutils.widgets.get("registered_model")
model_alias = dbutils.widgets.get("model_alias")

assert uc_catalog != "", "uc_catalog notebook parameter must be specified"
assert schema != "", "schema notebook parameter must be specified"
assert eval_table != "", "eval_table notebook parameter must be specified"
assert experiment != "", "experiment notebook parameter must be specified"
assert registered_model != "", "registered_model notebook parameter must be specified"
assert model_alias != "", "model_alias notebook parameter must be specified"

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

# DBTITLE 1,Create Evaluation Dataset
import mlflow.genai.datasets

try:
    eval_dataset = mlflow.genai.datasets.create_dataset(
        uc_table_name=f"{uc_catalog}.{schema}.{eval_table}",
    )
except:
    # Eval table already exists
    eval_dataset = mlflow.genai.datasets.get_dataset(
        uc_table_name=f"{uc_catalog}.{schema}.{eval_table}",
    )

print(f"Evaluation dataset: {uc_catalog}.{schema}.{eval_table}")

# COMMAND ----------

# DBTITLE 1,Get Reference Documentation
from utils.evaluation import get_reference_documentation

reference_docs = get_reference_documentation(uc_catalog, schema, eval_table, spark)

display(reference_docs)

# COMMAND ----------

# DBTITLE 1,Merge Reference Docs to Eval Dataset
eval_dataset.merge_records(reference_docs.limit(100))

# Preview the dataset
display(eval_dataset.to_df())

# COMMAND ----------

# DBTITLE 1,Load Model and Create Prediction Wrapper
import mlflow
import pandas as pd

# Set experiment
mlflow.set_experiment(experiment)

# Load the model
model_uri = f"models:/{uc_catalog}.{schema}.{registered_model}@{model_alias}"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Create a simple prediction wrapper that extracts the final answer
def predict_wrapper(question):
    """
    Simple prediction wrapper for evaluation.
    Takes a question string, returns the answer string.
    """
    # Format input for the model
    model_input = pd.DataFrame({
        "input": [[{"role": "user", "content": question}]]
    })
    response = loaded_model.predict(model_input)

    # Extract the final text response from the output
    # The output is a list of items (function calls, tool outputs, messages)
    # We need to find the last message with type='message' and role='assistant'
    for item in reversed(response['output']):
        if item.get('type') == 'message' and item.get('role') == 'assistant':
            # Content can be a list or a string
            content = item['content']
            if isinstance(content, list):
                # Get the last text content item
                for c in reversed(content):
                    if c.get('type') == 'text':
                        return c['text']
            elif isinstance(content, str):
                return content

    # Fallback to the original extraction method if the above doesn't work
    return response['output'][-1]['content'][-1]['text']

# COMMAND ----------

# DBTITLE 1,Test predict_wrapper with a sample question
# Test the wrapper with a sample question to inspect the response
test_question = "What is Databricks?"
print(f"Testing with question: {test_question}")
print("\n" + "="*80)

test_input = pd.DataFrame({
    "input": [[{"role": "user", "content": test_question}]]
})
test_response = loaded_model.predict(test_input)

print("Raw response structure:")
print(f"Response keys: {test_response.keys()}")
print(f"\nNumber of output items: {len(test_response['output'])}")
print("\nOutput items:")
for i, item in enumerate(test_response['output']):
    print(f"\n[{i}] Type: {item.get('type')}, Role: {item.get('role', 'N/A')}")
    if 'content' in item:
        content = item['content']
        if isinstance(content, list):
            print(f"    Content: list with {len(content)} items")
            for j, c in enumerate(content):
                if isinstance(c, dict):
                    print(f"      [{j}] {c.get('type', 'unknown')}: {str(c.get('text', c))[:100]}...")
                else:
                    print(f"      [{j}] {str(c)[:100]}...")
        else:
            print(f"    Content: {str(content)[:100]}...")

print("\n" + "="*80)
print(f"Extracted answer: {predict_wrapper(test_question)}")
print("="*80)

# COMMAND ----------

# DBTITLE 1,Define Evaluation Scorers
from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery, Safety, Guidelines

def get_scorers():
    return [
        RetrievalGroundedness(),  # Checks if response is grounded in retrieved data
        RelevanceToQuery(),  # Checks if response addresses the user's request
        Safety(),  # Checks for harmful or inappropriate content
        Guidelines(
            guidelines="""
            Response must be clear and professional.
            - Do not mention internal tools or functions used
            - Do not show intermediate reasoning steps
            - Provide direct, actionable answers
            """,
            name="response_quality",
        )
    ]

scorers = get_scorers()

# COMMAND ----------

# DBTITLE 1,Run Evaluation
import os
import warnings

# Suppress VectorSearch authentication notices during evaluation
os.environ["DATABRICKS_VECTOR_SEARCH_DISABLE_NOTICE"] = "1"
warnings.filterwarnings('ignore', message='.*notebook authentication token.*')

with mlflow.start_run(run_name="agent_evaluation"):
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_wrapper,
        scorers=scorers
    )

# COMMAND ----------

# Display results
print(f"Evaluation complete")
print(f"Metrics: {results.metrics}")