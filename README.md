  # agentops_dev

This directory contains an Agent project based on the default
[Databricks AgentOps Stacks](https://github.com/databricks/mlops-stacks),
defining a production-grade Agent pipeline for automated data preparation, agent development, and deploymant of a chatbot agent.
The "Getting Started" docs can be found at https://learn.microsoft.com/azure/databricks/dev-tools/bundles/mlops-stacks.

See the full pipeline structure below. The [AgentOps Stacks README](https://github.com/databricks/mlops-stacks/blob/main/Pipeline.md)
contains additional details on how Agent pipelines are tested and deployed across each of the dev, staging, prod environments below.


## Code structure
This project contains the following components:

| Component                  | Description                                                                                                                                                                                                                                                                                                                                             |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Agent Code                    | Example Agent project code and notebooks                                                                                                                                                                                                                                                                             |
| Agent Resources as Code | Agent pipeline resources (data preparation and development jobs with schedules, etc) configured and deployed through [Databricks CLI bundles](https://learn.microsoft.com/azure/databricks/dev-tools/cli/bundle-cli)                                                                                              |
| CI/CD                      | [GitHub Actions](https://github.com/actions) workflows  to test and deploy code and resources
                                |

contained in the following files:

```
agentops_dev        <- Root directory. Both monorepo and polyrepo are supported.
│
├── agentops_dev       <- Contains python code, notebooks and resources related to one project. 
│   │
│   ├── requirements.txt        <- Specifies Python dependencies for code.
│   │
│   ├── databricks.yml          <- databricks.yml is the root bundle file for the ML project that can be loaded by Databricks CLI bundles. It defines the bundle name, workspace URL and resource config component to be included.
│   │
│   ├── data_preparation        <- Retrieves, stores, cleans, and vectorizes source data that is then ingested into a Vector Search index.
│   │   │
│   │   ├── data_ingestion                             <- Databricks Documentation scraping retrieval and storage.
│   │   │
│   │   ├── data_preprocessing                         <- Documentation cleansing and vectorization.
│   │   │
│   │   ├── vector_search                              <- Vector Search and index creation and ingestion.
│   │
│   │
│   ├── agent_development       <- Creates, registers, and evaluates the agent.
│   │   │
│   │   ├── agent                                      <- LangGraph Agent creation.
│   │   │
│   │   ├── agent_evaluation                           <- Databricks Agent llm-as-a-judge evaluation.
│   │
│   ├── agent_deployment        <- Deploys agent serving and contains a Databricks Apps front end interface.
│   │   │
│   │   ├── chat_interface_deployment                  <- Databricks App front end interface for end users.
│   │   │
│   │   ├── model_serving                              <- Model serving endpoint for the Agent.
│   │
│   │
│   ├── tests                   <- Tests for the Agent project.
│   │
│   ├── resources               <- Agent resource (Agent jobs, MLflow models) config definitions expressed as code, across dev/staging/prod/test.
│       │
│       ├── data-preparation-resource.yml              <- Agent resource config definition for data preparation and vectorization.
│       │
│       ├── agent-resource-workflow-resource.yml       <- Agent resource config definition for agent development, evaluation, and deployment.
│       │
│       ├── app-deployment-resource.yml                <- Agent resource config definition for launching the Databricks App frontend.
│       │
│       ├── agents-artifacts-resource.yml                  <- Agent resource config definition for model and experiment.
│
├── .github                     <- Configuration folder for CI/CD using GitHub Actions.  The CI/CD workflows deploy resources defined in the `./resources/*` folder with Databricks CLI bundles.
│
├── docs                        <- Contains documentation for the repo.
│
├── cicd.tar.gz                 <- Contains CI/CD bundle that should be deployed by deploy-cicd.yml to set up CI/CD for projects.
```

## Using this repo

The table below links to detailed docs explaining how to use this repo for different use cases.


This project comes with example Agent code to develop, evaluate and deploy a chatbot Agent that answers question regarding Databricks, retrieving relevant information from Databricks documentation.


If you're a data scientist just getting started with this repo for a brand new Agent project, we recommend adapting the provided example code to your Agent problem. Then making and testing Agent code changes on Databricks or your local machine. Follow the instructions from the [project README](./agentops_dev/README.md).
 

When you're ready to deploy production training/inference
pipelines, ask your ops team to follow the [MLOps setup guide](docs/agentops-setup.md) to configure CI/CD and deploy production pipelines.

After that, follow the [pull request guide](docs/pull-request.md)
 and [agent resource config guide](agentops_dev/resources/README.md)  to propose, test, and deploy changes to production Agent code (e.g. update model parameters) or pipeline resources (e.g. use a larger instance type for model training) via pull request.

| Role                          | Goal                                                                         | Docs                                                                                                                                                                |
|-------------------------------|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Data Scientist                | Get started writing Agent code for a brand new project                          | [project README](./agentops_dev/README.md) |
| AgentOps / DevOps                | Set up CI/CD for the current Agent project   | [AgentOps setup guide](docs/agentops-setup.md)                                                                                                                            |
| Data Scientist                | Update production Agent code for an existing project | [pull request guide](docs/pull-request.md)                                                                                                                    |
| Data Scientist                | Modify production model Agent resources, e.g. data preparation or agent development jobs  | [Agent resource config guide](agentops_dev/resources/README.md)  |

## Setting up CI/CD
This stack comes with a workflow to set up CI/CD for projects that can be found in

`.github/workflows/deploy-cicd.yml`.


To set up CI/CD for projects that were created through AgentOps Stacks with the `Project_Only` parameter, run the above mentioned workflow, specifying the `project_name` as a parameter. For example, for the monorepo case:

1. Setup your repository by initializing Agent Ops Stacks via Databricks CLI with the `CICD_and_Project` or `CICD_Only` parameter.
2. Follow the [AgentOps Setup Guide](./docs/agentops-setup.md) to setup authentication and get the repo ready for CI/CD.
3. Create a new project by initializing AgentOps Stacks again but this time with the `Project_Only` parameter.
4. Run the `deploy-cicd.yml` workflow with the `project_name` parameter set to the name of the project you want to set up CI/CD for.


NOTE: This project has already been initialized with an instantiation of the above workflow, so there's no
need to run it again for project `agentops_dev`.
