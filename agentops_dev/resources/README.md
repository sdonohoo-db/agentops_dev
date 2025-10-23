# Databricks Agent Resource Configurations
[(back to project README)](../README.md)

## Table of contents
* [Intro](#intro)
* [Local development and dev workspace](#local-development-and-dev-workspace)
* [Develop and test config changes](#develop-and-test-config-changes)

## Intro

### databricks CLI bundles
AgentOps Stacks resources are configured and deployed through [Databricks CLI bundles](https://learn.microsoft.com/azure/databricks/dev-tools/cli/bundle-cli).
The bundle setting file must be expressed in YAML format and must contain at minimum the top-level bundle mapping.

The Databricks CLI bundles top level is defined by file `agentops_dev/databricks.yml`.
During Databricks CLI bundles deployment, the root config file will be loaded, validated and deployed to workspace provided by the environment together with all the included resources.

Agent Resource Configurations in this directory:
 - data preparation workflow (`agentops_dev/resources/data-preparation-resource.yml`)
 - agent development and deployment workflow (`agentops_dev/resources/agent-resource.yml`)
 - app deployment workflow (`agentops_dev/resources/app-deployment-resource.yml`)
 - model definition, experiment, and app definition (`agentops_dev/resources/agents-artifacts-resource.yml`)


### Deployment Config & CI/CD integration
The agent resources can be deployed to a Databricks workspace based on the Databricks CLI bundles deployment config. Deployment configs of different deployment targets share the general agent resource configurations with added ability to specify deployment target specific values (workspace URI, model name, jobs notebook parameters, etc).

NOTE: This project was not setup with CI/CD workflows. You can setup CI/CD with a new initialization of AgentOps Stacks. The rest of this section only applies if you are using a monorepo setup with CI/CD previously or have setup CI/CD otherwise.

When you initialize the stack, we set the catalog name in the `agentops_dev/databricks.yml`, so we expect a catalog of the same name in each environment. I

If you want to use different catalog names, please set the variable `uc_catalog` under each target environment: 

```
targets:
  dev:
    variables:
      uc_catalog: 
        description: Unity Catalog used to store data and artifacts.
        default: <insert-different-catalog-name>

```

| Deployment Target | Description                                                                                                                                                                                                                           | Databricks Workspace |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| dev         | The `dev` deployment target is used to deploy resources to development workspace with `dev` configs. The config is for project development purposes.                                                           | dev workspace        | dev-agentops_dev-model     | /dev-agentops_dev-experiment     |
| staging     | The `staging` deployment target is part of the CD pipeline. Latest main content will be deployed to staging workspace with `staging` config.                                                             | staging workspace    | staging-agentops_dev-model | /staging-agentops_dev-experiment |
| prod        | The `prod` deployment target is part of the CD pipeline. Latest release content will be deployed to prod workspace with `prod` config.                                                                      | prod workspace       | prod-agentops_dev-model    | /prod-agentops_dev-experiment    |
| test        | The `test` deployment target is part of the CI pipeline. For changes targeting the main branch, upon making a PR, an integration test will be triggered and agent resources deployed to the staging workspace defined under `test` deployment target. | staging workspace    | test-agentops_dev-model    | /test-agentops_dev-experiment    |

During code development, you can deploy local resource configurations together with code to the Databricks workspace to run the ingestion, development, or deployment pipelines. The deployment will use `dev` config by default.

You can open a PR (pull request) to modify code or the resource config against main branch.
The PR will trigger Python unit tests, followed by an integration test executed on the staging workspace, as defined under the `test` environment resource. 

Upon merging a PR to the main branch, the main branch content will be deployed to the staging workspace with `staging` environment resource configurations.

Upon merging code into the release branch, the release branch content will be deployed to prod workspace with `prod` environment resource configurations.

## Local development and dev workspace

### Set up authentication

To set up the Databricks CLI using a Databricks personal access token, take the following steps:

1. Follow [Databricks CLI](https://learn.microsoft.com/azure/databricks/dev-tools/cli/databricks-cli) to download and set up the Databricks CLI locally.
2. Complete the `TODO` in `agentops_dev/databricks.yml` to add the dev workspace URI under `targets.dev.workspace.host`.
3. [Create a personal access token](https://learn.microsoft.com/azure/databricks/dev-tools/auth/pat)
  in your dev workspace and copy it.
4. Set an env variable `DATABRICKS_TOKEN` with your Databricks personal access token in your terminal. For example, run `export DATABRICKS_TOKEN=dapi12345` if the access token is dapi12345.
5. You can now use the Databricks CLI to validate and deploy resource configurations to the dev workspace.

Alternatively, you can use the other approaches described in the [Databricks CLI](https://learn.microsoft.com/azure/databricks/dev-tools/cli/databricks-cli) documentation to set up authentication. For example, using your Databricks username/password, or seting up a local profile.

### Validate and provision agent resource configurations
1. After installing the Databricks CLI and creating the `DATABRICKS_TOKEN` env variable, change to the `agentops_dev` directory.
2. Run `databricks bundle validate` to validate the Databricks resource configurations. 
3. Run `databricks bundle deploy` to provision the Databricks resource configurations to the dev workspace. The resource configurations and your code will be copied together to the dev workspace. The defined resources such as Databricks Workflows, Registered Model and MLflow Experiment will be provisioned according to the config files under `agentops_dev/resources`.
4. Go to the Databricks dev workspace, check the defined model, experiment and workflows status, and interact with the created workflows.

### Destroy resource configurations
After development is done, you can run `databricks bundle destroy` to destroy (remove) the defined Databricks resources in the dev workspace. 

Any model version will prevent the model from being deleted. Please update the version stage to `None` or `Archived` before destroying the ML resources.

In addition, currently, there are a number of assets not managed by the bundle that cannot be deleted by the `destroy` command, including: 

- Agent deployment. Additionally, assets created by the deployment (the Feedback registered model, the CPU endpoint for the agent and feedback model, and the payload table). 
- Data assets, including the volume with Databricks documentation and the three Delta tables. 
- Vector Search assets (the endpoint and index). 
- UC Functions.

## Develop and test config changes

### Databricks CLI bundles schema overview
To get started, open `agentops_dev/resources/batch-inference-workflow-resource.yml`.  The file contains the agent resource definition of a data ingestion job, like:

```$xslt
resources:
  jobs:
    data_preprocessing_job:
      name: ${bundle.target}-agentops_dev-data-preprocessing-job
      tasks:
        - task_key: RawDataIngest
          notebook_task:
            notebook_path: ../data_preparation/data_ingestion/notebooks/DataIngestion.py
            base_parameters:
              # TODO modify these arguments to reflect your setup.
              uc_catalog: ${var.uc_catalog}
              schema: ${var.schema}
              raw_data_table: ${var.raw_data_table}
              data_source_url: https://docs.databricks.com/en/doc-sitemap.xml
              # git source information of current ML resource deployment. It will be persisted as part of the workflow run
              git_source_info: url:${bundle.git.origin_url}; branch:${bundle.git.branch}; commit:${bundle.git.commit}
              ...
```

The example above defines a Databricks job with name `${bundle.target}-agentops_dev-data-preprocessing-job`
that runs the notebook under `agentops_dev/data_preparation/data_ingestion/notebooks/DataIngestion.py` to ingest documents from our website. 

As this is running on serverless, there is no need for cluster definitions. If you are deploying to a non-serverless workspace, please reference the [MLOps Stacks Resource README](https://github.com/databricks/mlops-stacks/blob/main/template/%7B%7B.input_root_dir%7D%7D/%7B%7Btemplate%20%60project_name_alphanumeric_underscore%60%20.%7D%7D/resources/README.md.tmpl).

We specify a `data_preprocessing_job` under `resources/jobs` to define a Databricks workflow with internal key `data_preprocessing_job` and job name `{bundle.target}-agentops_dev-data-preprocessing-job`.
The workflow contains a single task with task key `data_preprocessing_job`. The task runs notebook `agentops_dev/data_preparation/data_ingestion/notebooks/DataIngestion.py` with provided parameters `uc_catalog`, `schema`, `raw_data_table`, and `data_source_url` passing to the notebook.
After setting up Databricks CLI, you can run command `databricks bundle schema` to learn more about Databricks CLI bundles schema.

The notebook_path is the relative path starting from the resource yaml file.

### Environment config based variables
The `${bundle.target}` will be replaced by the environment config name during the bundle deployment. For example, during the deployment of a `test` environment config, the job name will be
`test-agentops_dev--data-preprocessing-job`. During the deployment of the `staging` environment config, the job name will be
`staging-agentops_dev--data-preprocessing-job`.


To use different values based on different environment, you can use bundle variables based on the given target, for example,
```$xslt
variables:
  raw_data_table: 
    description: The table name to be used for storing the raw data.
    default: input_table

targets:
  dev:
    variables:
      raw_data_table: dev_table
  test:
    variables:
      raw_data_table: test_table

resources:
  jobs:
    data_preprocessing_job:
      name: ${bundle.target}-agentops_dev-data-preprocessing-job
      tasks:
        - task_key: RawDataIngest
          notebook_task:
            notebook_path: ../data_preparation/data_ingestion/notebooks/DataIngestion.py
            base_parameters:
              # TODO modify these arguments to reflect your setup.
              uc_catalog: ${var.uc_catalog}
              schema: ${var.schema}
              raw_data_table: ${var.raw_data_table}
              data_source_url: https://docs.databricks.com/en/doc-sitemap.xml
              # git source information of current ML resource deployment. It will be persisted as part of the workflow run
              git_source_info: url:${bundle.git.origin_url}; branch:${bundle.git.branch}; commit:${bundle.git.commit}
              ...
```
The `data_preprocessing_job` notebook parameter `raw_data_table` is using a bundle variable `raw_data_table` with default value "input_table".
The variable value will be overwritten with "dev_table" for `dev` environment config and "test_table" for `test` environment config:
- during deployment with the `dev` environment config, the `input_table_name` parameter will get the value "dev_table"
- during deployment with the `staging` environment config, the `input_table_name` parameter will get the value "input_table"
- during deployment with the `prod` environment config, the `input_table_name` parameter will get the value "input_table"
- during deployment with the `test` environment config, the `input_table_name` parameter will get the value "test_table"

### Test config changes
To test out a config change, simply edit one of the fields above.

Then follow [Local development and dev workspace](#local-development-and-dev-workspace) to deploy the change to the dev workspace.
Alternatively you can open a PR. Continuous integration will then validate the updated config and deploy tests to the to staging workspace.

[Back to project README](../README.md)
