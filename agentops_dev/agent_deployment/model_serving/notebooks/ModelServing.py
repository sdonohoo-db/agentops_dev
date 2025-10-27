# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# DBTITLE 1,Model Serving Pipeline - Overview
##################################################################################
# Model Serving
# 
# Helper notebook to serve the model on an endpoint. This notebook is run
# after the ModelDeployment.py notebook as part of a multi-task job, in order to serve the model
# on an endpoint stage after transitioning the latest version.
#
# Parameters:
# * uc_catalog (required)                   - Name of the Unity Catalog
# * schema (required)                       - Name of the schema inside Unity Catalog
# * registered_model (required)             - Name of the model registered in mlflow
# * model_alias (required)                  - Model alias to deploy
# * scale_to_zero (required)                - Specify if the endpoint should scale to zero when not in use.
# * workload_size (required)                - Specify  the size of the compute scale out that corresponds with the number of requests this served
#                                             model can process at the same time. This number should be roughly equal to QPS x model run time.
# * agent_serving_endpoint (required)       - Name of the agent serving endpoint to deploy
#
# Widgets:
# * Unity Catalog: Text widget to input the name of the Unity Catalog
# * Schema: Text widget to input the name of the database inside the Unity Catalog
# * Registered model name: Text widget to input the name of the model to register in mlflow
# * Model Alias: Text widget to input the model alias to deploy
# * Scale to zero: Whether the clusters should scale to zero (requiring more time at startup after inactivity)
# * Workload Size: Compute that matches estimated number of requests for the endpoint
# * Agent model serving endpoint: Text widget to input the name of the model serving endpoint to deploy
#
# Usage:
# 1. Set the appropriate values for the widgets.
# 2. Add members that you want to grant access to for the review app to the user_list.
# 3. Run to deploy endpoint.
#
##################################################################################

# COMMAND ----------

# DBTITLE 1,Install Prerequisites
# Install prerequisite packages
%pip install -r ../../../agent_development/agent_requirements.txt

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
# Scale to zero
dbutils.widgets.dropdown("scale_to_zero", "True", ["True", "False"], "Scale to zero")
# Workdload size
dbutils.widgets.dropdown("workload_size", "Small", ["Small", "Medium", "Large"], "Workload Size")
# Agent serving endpoint
dbutils.widgets.text(
    "agent_serving_endpoint",
    "chatbot_model_serving_endpoint",
    label="Agent serving endpoint",
)

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Define Input Variables
uc_catalog = dbutils.widgets.get("uc_catalog")
schema = dbutils.widgets.get("schema")
registered_model = dbutils.widgets.get("registered_model")
model_alias = dbutils.widgets.get("model_alias")
scale_to_zero = bool(dbutils.widgets.get("scale_to_zero"))
workload_size = dbutils.widgets.get("workload_size")
agent_serving_endpoint = dbutils.widgets.get("agent_serving_endpoint")

assert uc_catalog != "", "uc_catalog notebook parameter must be specified"
assert schema != "", "schema notebook parameter must be specified"
assert registered_model != "", "registered_model notebook parameter must be specified"
assert model_alias != "", "model_alias notebook parameter must be specified"
assert scale_to_zero != "", "scale_to_zero notebook parameter must be specified"
assert workload_size != "", "workload_size notebook parameter must be specified"
assert agent_serving_endpoint != "", "agent_serving_endpoint notebook parameter must be specified"

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
# DBTITLE 1,Review Instructions
instructions_to_reviewer = f"""### Instructions for Testing the our Chatbot assistant

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement.

1. **Variety of Questions**:
   - Please try a wide range of questions that you anticipate the end users of the application will ask. This helps us ensure the application can handle the expected queries effectively.

2. **Feedback on Answers**:
   - After asking each question, use the feedback widgets provided to review the answer given by the application.
   - If you think the answer is incorrect or could be improved, please use "Edit Answer" to correct it. Your corrections will enable our team to refine the application's accuracy.

3. **Review of Returned Documents**:
   - Carefully review each document that the system returns in response to your question.
   - Use the thumbs up/down feature to indicate whether the document was relevant to the question asked. A thumbs up signifies relevance, while a thumbs down indicates the document was not useful.

Thank you for your time and effort in testing our assistant. Your contributions are essential to delivering a high-quality product to our end users."""

# COMMAND ----------
# DBTITLE 1,Deploy Agent

from databricks import agents
from mlflow import MlflowClient

client = MlflowClient()

model_name = f"{uc_catalog}.{schema}.{registered_model}"
model_version = int(client.get_model_version_by_alias(model_name, model_alias).version)

print(f"Deploying model {model_name} version {model_version} to endpoint {agent_serving_endpoint}")

existing_deployments = agents.get_deployments(endpoint_name=agent_serving_endpoint)
if len(existing_deployments) == 0:
    agents.deploy(
        model_name=model_name,
        model_version=model_version,
        endpoint_name=agent_serving_endpoint
    )
    print(f"Deployed model {model_name} version {model_version} to endpoint {agent_serving_endpoint}")
else:
    print(f"A model is already deployed to endpoint {agent_serving_endpoint}. Skipping deployment.")

# Set review instructions
agents.set_review_instructions(model_name, instructions_to_reviewer)

# COMMAND ----------
# DBTITLE 1,Wait for Endpoint to be Ready

from utils import wait_for_model_serving_endpoint_to_be_ready
wait_for_model_serving_endpoint_to_be_ready(agent_serving_endpoint)

# COMMAND ----------

# DBTITLE 1,Grant Permissions
#TODO grant your stakeholders permissions to use the Review App
# user_list = ["firstname.lastname@company.com"]

# Set the permissions.

# agents.set_permissions(model_name=model_name, users=user_list, permission_level=agents.PermissionLevel.CAN_QUERY)

# print(f"Share this URL with your stakeholders: {deployment_info.review_app_url}")

# COMMAND ----------
# DBTITLE 1,Test Endpoint

from mlflow.deployments import get_deploy_client

client = get_deploy_client()
input_example = {
    "input": [{"role": "user", "content": "What is MLflow?"}],
    "databricks_options": {"return_trace": True},
}

response = client.predict(endpoint=agent_serving_endpoint, inputs=input_example)

print(response['output'])

