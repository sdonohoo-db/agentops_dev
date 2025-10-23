# AgentOps Setup Guide
[(back to main README)](../README.md)

## Table of contents
* [Intro](#intro)
* [Create a hosted Git repo](#create-a-hosted-git-repo)
* [Configure CI/CD](#configure-cicd---github-actions)
* [Merge PR with initial ML code](#merge-a-pr-with-your-initial-ml-code)
* [Create release branch](#create-release-branch)

* [Deploy ML resources and enable production jobs](#deploy-ml-resources-and-enable-production-jobs)
* [Next steps](#next-steps)

## Intro
This page explains how to productionize the current project, setting up CI/CD and agent resource deployment, and deploying agent development and deployment.

After following this guide, data scientists can follow the [Pull Request](pull-request.md) guide to make changes to agent code or deployed jobs.

## Create a hosted Git repo
Create a hosted Git repo to store project code, if you haven't already done so. From within the project
directory, initialize Git and add your hosted Git repo as a remote:
```
git init --initial-branch=main
```

```
git remote add upstream <hosted-git-repo-url>
```

Commit the current `README.md` file and other docs to the `main` branch of the repo, to enable forking the repo:
```

git add README.md docs .gitignore agentops_dev/resources/README.md
git commit -m "Adding project README"

git push upstream main
```

## Configure CI/CD - GitHub Actions

### Prerequisites
* You must be an account admin to add service principals to the account.
* You must be a Databricks workspace admin in the staging and prod workspaces. 
  Verify that you're an admin by viewing the
  [staging workspace admin console](https://adb-2350455894922354.14.azuredatabricks.net#setting/accounts) and
  [prod workspace admin console](https://adb-2350455894922354.14.azuredatabricks.net#setting/accounts). 
  If the admin console UI loads instead of the Databricks workspace homepage, you are an admin.

### Set up authentication for CI/CD
#### Set up Service Principal

To authenticate and manage agent resources created by CI/CD, 
[service principals](https://learn.microsoft.com/azure/databricks/administration-guide/users-groups/service-principals)
for the project should be created and added to both staging and prod workspaces. Follow
[Add a service principal to your Azure Databricks account](https://learn.microsoft.com/azure/databricks/administration-guide/users-groups/service-principals#--add-a-service-principal-to-your-azure-databricks-account)
and [Add a service principal to a workspace](https://learn.microsoft.com/azure/databricks/administration-guide/users-groups/service-principals#--add-a-service-principal-to-a-workspace)
for details.

For your convenience, we also have Terraform modules that can be used to [create](https://registry.terraform.io/modules/databricks/mlops-azure-project-with-sp-creation/databricks/latest) or [link](https://registry.terraform.io/modules/databricks/mlops-azure-project-with-sp-linking/databricks/latest) service principals.



#### Configure Service Principal (SP) permissions 
When you initialize the stack, we set the catalog name in the `databricks.yml`, so we expect a catalog of the same name in each environment. I

If you want to use different catalog names, please set `uc_catalog` differently under each target environment: 

```
targets:
  dev:
    variables:
      uc_catalog: 
        description: Unity Catalog used to store data and artifacts.
        default: <insert-different-catalog-name>

```

The SP must have proper permission in each respective environment and the catalog for the environments.

For the integration tests and workflows, the SP must have permissions to read + write to the specified schema and create experiment and models. 
i.e. for each environment:
- USE_CATALOG
- USE_SCHEMA
- MODIFY
- CREATE_MODEL
- CREATE_TABLE
- CREATE_VOLUME

#### Set secrets for CI/CD

After creating the service principals and adding them to the respective staging and prod workspaces, refer to
[Manage access tokens for a service principal](https://learn.microsoft.com/azure/databricks/administration-guide/users-groups/service-principals#--manage-access-tokens-for-a-service-principal)
and [Get Azure AD tokens for service principals](https://learn.microsoft.com/azure/databricks/dev-tools/api/latest/aad/service-prin-aad-token)
to get your service principal credentials (tenant id, application id, and client secret) for both the staging and prod service principals, and [Encrypted secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
to add the following secrets to GitHub:
- `PROD_AZURE_SP_TENANT_ID`
- `PROD_AZURE_SP_APPLICATION_ID`
- `PROD_AZURE_SP_CLIENT_SECRET`
- `STAGING_AZURE_SP_TENANT_ID`
- `STAGING_AZURE_SP_APPLICATION_ID`
- `STAGING_AZURE_SP_CLIENT_SECRET`
- `WORKFLOW_TOKEN` : [Github token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic) with workflow permissions. This secret is needed for the Deploy CI/CD Workflow.
Be sure to update the [Workflow Permissions](https://docs.github.com/en/actions/security-guides/automatic-token-authentication#modifying-the-permissions-for-the-github_token) section under Repo Settings > Actions > General to allow `Read and write permissions`.


### Setting up CI/CD workflows
After setting up authentication for CI/CD, you can now set up CI/CD workflows. We provide a [Deploy CICD workflow](../.github/workflows/deploy-cicd.yml) that can be used to generate the other CICD workflows mentioned below for projects. 
This workflow is manually triggered with `project_name` as parameter. This workflow will need to be triggered for each project to set up its set of CI/CD workflows that can be used to deploy resources and run jobs in the staging and prod workspaces. 
These workflows will be defined under `.github/workflows`.

If you want to deploy CI/CD for an initialized project (`Project-Only` AgentOps Stacks initialization), you can manually run the `deploy-cicd.yml` workflow from the [Github Actions UI](https://docs.github.com/en/actions/using-workflows/manually-running-a-workflow?tool=webui) once the project code has been added to your main repo. The workflow will create a pull request with all the changes against your main branch. Review and approve it to commit the files to deploy CI/CD for the project. 



## Merge a PR with your initial agent code
Create and push a PR branch adding the agent code to the repository.

```
git checkout -b add-agent-code
git add .
git commit -m "Add agent code"
git push upstream add-agent-code
```

Open a PR from the newly pushed branch. CI will run to ensure that tests pass
on your initial agent code. Fix tests if needed, then get your PR reviewed and merged.
After the pull request merges, pull the changes back into your local `main`
branch:

```
git checkout main
git pull upstream main
```

## Create release branch
Create and push a release branch called `release` off of the `main` branch of the repository:
```
git checkout -b release main
git push upstream release
git checkout main
```

Your production jobs will pull the agent code against this branch, while your staging jobs will pull the agent code against the `main` branch. Note that the `main` branch will be the source of truth for agent resource configs and CI/CD workflows.

For future code changes, iterate against the `main` branch and regularly deploy your code from staging to production by merging code changes from the `main` branch into the `release` branch.

## Deploy agent resources and enable production jobs
Follow the instructions in [agentops_dev/resources/README.md](../agentops_dev/resources/README.md) to deploy agent resources and production jobs.

## Next steps
After you configure CI/CD and deploy training & inference pipelines, notify data scientists working
on the current project. They should now be able to follow the
[pull request guide](pull-request.md) and 
[Agent resource config guide](../agentops_dev/resources/README.md)  to propose, test, and deploy
Agent code and pipeline changes to production.