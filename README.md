<center> <h1 style=color:red> MLOps </h1> </center>

<center> <h1 style=color:green> Introduction to MLOps </h1> </center>

<center> <h3> What is MLOps? </h3> </center>

<center> <h4> ML + OPs = Machine Learning + Operations </h4> </center>

1. A set of principles and practices to standardize and streamline the machine learning lifecycle management.
2. Integration between development and operational processes.
3. Teams collaborate to build, automate, test, and monitor the machine learning pipelines.

<center> <h1 style=color:brown> Machine Learning Lifecycle </h1> </center>

### 1. Business Understanding/Requirement Gathering

### 2. Data Acquisition
i. **Identify data sources**
ii. **Identify data formats**
iii. **Make sources connections**
iv. **Create data lake**

### 3. Feature Engineering
i. **Data Wrangling**
> * Clean the data.

ii. **Data Transformations**
> * Transform the data (Joins, String functions, mathematical operations, etc.)

iii. **Data Pre-processing**

iv. **Data Visualization**
> * Visualize data in graph, heatmaps, bar charts, etc.

v. **Exploratory Data Analysis**
> * EDA- initialize investigation of data to discover patterns, detect outliers, spot anomalies, test hypothesis.

### 4. Data Modelling
i. **Model Selection**
> * Select the algorithm as per use case.

ii. **Model Building**
> * Split the data.

iii. **Model Training**
> * Train the model on bigger chunk.

iv. **Model Validation/Testing**
> * Validate and test the trained model on the remaining chunk.
> * Try different combinations of data, algorithm, features.
> * Hyperparameter tuning.

### 5. Model Deployment
Model deployment in form of webservices, API.

### 6. Monitoring
Monitoring the model.

### 7. Model Retraining
Retraining on new data.

<center> <h1 style=color:brown> Challenges in  Machine Learning Projects </h1> </center>

### ***Data Scientists***
>  i. **Data Acquisition**
ii. **Data Pre-processing**
iii. **Model Building**
iv. **Model Training**
v. **Model Validation**

### ***Operational Team***
> i. **Package**
ii. **Compile**
iii. **Deploy**
iv. **Release**
v. **Monitoring**

* What are the required dependencies?
* Package requires?
* What feature parameter model is trained on?
* What variable represents model output values.
* What is the data format to be feed to model?
* Is model production ready?

<center> <h1 style=color:brown> Activities to Productionize a Model </h1> </center>

## Build & Test Locally

### 1. **Package**
* Compile the code
* Resolve dependencies
* Run Scripts

### 2. Performance
* Scaling out to train huge data
* Model performance for huge data
* Load balancing
* Add parallelism
* Data partitioning
* Take GPU support
* Model prediction speed

### Instrument
* Versioning, repo management, security, monitoring
* Versioning of algorithm + data + feature parameters + environments
* Reproducibility

### Automation
* Lot of mutual tasks
* Continuous Training required

<center> <h1 style=color:brown> Technical Debt in Machine Learning </h1> </center>

<table style="border-spacing: 20px; width: 100%;">
  <tr>
    <td style="background-color: lightblue;">Configuration</td>
    <td style="background-color: lightblue;">Data Collection</td>
    <td style="background-color: lightblue;">Versioning</td>
    <td style="background-color: lightblue;">Resource Management</td>
    <td style="background-color: lightblue;">Serving Infrastructure</td>
  </tr>
  <tr>
    <td rowspan="2" style="background-color: lightblue;"></td>
    <td rowspan="2" style="background-color: lightblue;">Feature Extraction</td>
    <td style="background-color: yellow;">ML Code</td>
    <td style="background-color: lightblue;">Analysis Tools</td>
    <td rowspan="2" style="background-color: lightblue;"></td>
  </tr>
  <tr>
    <td style="background-color: lightblue;">Process Management Tools</td>
    <td style="background-color: lightblue;">Monitoring</td>
  </tr>
</table>

<center> <h1 style=color:brown> How MLOPs Address the Challenges </h1> </center>

**MLOPs** - A culture with set of principles, guidelines defined in a machine learning world to seamlessly integrate/automate the development phase with the operational phase.

### Define Standards/Principles

1. **Transaction Friction**
> * Use notebook templated that define common functionality e.g. database connection template, fetching data, etc.
> * Proper documentation

2. **Version Control System**
> * Version control system for code, data, environments and artifacts.

3. **Performance**
> * Leverage distributed computing and containerization tools e.g. Docker, Kubernetes.

4. **Automation**
> * Building workflow, CI/CD pipelines.
> * Mlops is pipeline centric. Put pipelines in production that putting a mode.

5. **Monitoring**
> * Powerful innovative monitoring tools to monitor data, features, distribution, latency, uptime, memory utilization, etc.
> * Monitoring tools e.g. Prometheus, Grafana.

<center> <h1 style=color:green> Introduction to MLflow </h1> </center>

***MLflow*** is an open-source platform for managing the ***end-to-end machine learning lifecycle*** including experimentation, reproducibility, deployment, and a central model registry. 

### MLflow Components

1. ***Tracking :*** Track experiments to record and compare parameters, results.

* Allows to track experiments by **recording the parameters, code versions, metrics and results** of experiments.
* Allows to **log and track** machine learning experiments in **a central place** using Python, REST, R and Java APIs.
* Provides a **web-based user interface** for exploring runs, visualizing metrics & parameters, and comparing experiments.
* Support **different tracking servers** including local and remote servers, making it easy to scale and share experiments across different terms.

2. ***Projects :*** Packages code to ensure reusability and reproducibility.
* Designed to **simplify the process** of packaging, reproducibility, and sharing of machine learning code.
* Allows users to package their code and dependencies into a **reproducible format** which is **easier to share** and **reproduce experiments** across different environments.
* **MLproject file** to describe the project environment, parameters, entry points and a set of files containing the machine learning code and data.

3. ***Models :*** Provides a standard unit for packaging models.
* Design to **streamline and simplify** the process of **deploying machine learning models** to **different environments**.
* Allows users to **package their trained models into a standard format** which is supported by many downstram tools in the pipeline.
* Saves the model in a number od distinct **"flavors"**. supports many libraries (as model flavors) including scikit-learn, Tensorflow, PyTorch, Keras, Spark Mllib, etc.
* Can be deployed using different deployment options such as **REST APIs, Docker containers, and Serverless functions**.

4. ***Registry :*** Central model store for model versioning, stage transitions and annotations, etc.
* **Centralized repository** for managing machine learning models, versions of these models, and metadata.
* Provides users with a **set of APIs and a user interface** to collaboratively manage the Model's entire lifecycle.
* **Search Interface** allows users to search for models by name, metadata, and other criteria, making it easy to find the right model quickly.
* Allows users to **add metadata** to each model version like person name, creation data, description, etc. All of this helps in finding your model quickly.

### Other Bits of MLflow

1. **Language Agnostic**
> * Modular API-first approach.
> * Can use it with ML library, and in any programming language.

2. **Compatibility**
> * Compatible with numerous libraries, such as Tensorflow, PyTorch, Keras, Apache spark, scikit-learn, etc.

3. **Integration**
> * Put the model to production, in the form Docker containers, Kubernetes cluster, Apache spark, etc.

4. **Creation**
> * Created by Databricks in June 2018.


<center> <h1 style=color:green> Logging Functions </h1> </center>

***`set_tracking uri()`*** : **Set** the default tracking URI of choice for current run.

*Parameters :*
* *`<uri>`*  *Location* where files shall be stored.
* *`Empty string`* $\rightarrow$ MLflow will automatically create a folder **mlruns** and stores all of your tracks in it.
* *`Folder Name`* $\rightarrow$ Provide the folder name of your choice using ***"./<name_of_ur_choice>"***

***`get_tracking uri()`*** : **Get** the tracking URI.

*Parameters :*
* *`<None>`*

***`create_experiment()`*** : Create a fresh **new** experiment.

*Parameters :*

* *`Name`* $\rightarrow$ Specify the **custom name** of your experiment. Name must be **unique and case sensitive**.
* *`artifact_location`* $\rightarrow$ Location to store the **artifacts folder in a specific location**.

* *`Tags`* $\rightarrow$ Specify tags on the experiment as **dictionary of key value pairs**.

*Returns :*

**String ID** of the created experiment.

***`set_experiment()`*** : Utilize an **existing experiment** to record runs.

*Parameters :*

* *`experiment_name`* $\rightarrow$ Name of **already existing experiment** into which runs shall be launched. If the experiment **doen not exist, a new experiment** which the specified name will get cheated.

* *`experiment_id`* $\rightarrow$ Id of **already existing experiment** into which new runs shall be launched.

*Returns :*

An instance of **mlflow.entities,Experiment** representing the new active experiment.

***`start_run()`*** : Start a **new MLFlow run** in an experiment under which all the entities will be logged.

*Parameters :*

* *`run_id`* $\rightarrow$  Specify the `run_id` argument when you **want to utilize the existing run**.

* *`experiment_id`* $\rightarrow$ Specify  the ID of the experiment under which the current run must be created. **Can only use**, if and only if the **run_id isn't specified**.

* *`run_name`* $\rightarrow$  Specify name of the new run. **Can only** be used if you have **not specified the run_id**.

* *`nested`* $\rightarrow$ To set any **run as nested run**, specify this argument as **True**.

* *`tags`* $\rightarrow$ Dictionary of string key value pairs to **set tags to the new/existing run**.

* *`description`* $\rightarrow$ Populates the description box of the new/existing run.

*Returns :*
Returns **mlflow.ActiveRun** object that acts as a context manager wrapping the run's state.


```plaintext

set_experiment()
    /\
    ||
MLFLOW_EXPERIMENT_NAME environment variable
    /\
    ||
MLFLOW_EXPERIMENT_ID environment variable
    /\
    ||
Default experiment name
````

***`active_run()`*** : Returns the run which is **currently active now**.

*Parameters :*
* *`None`*
*Return :*
Run object

***`last_active_run()`*** : Gets the **moat recent active** run.

*Parameters :*
* *`None`*
*Return :*
Run object

<center> <h2> Log Multiple Parameters and Metrics Functions </h2> </center>

***Core Logging Functions :***

| **Single Function**     | Work                                                 |  **Multi-Function**    | Work                  |
| ----------------------- | ---------------------------------------------------- | ---------------------- | --------------------- |
| `mlflow.log_param()`    | Logs a **single** hyperparameter as key-value pair.  | `mlflow.log_params()`  | Logs **multiple** hyperparameter as key-value pair. |
| `mlflow.log_metric()`   | Logs a **single** metric as key-value pair.          | `mlflow.log_metrics()`  | Logs **multiple** metric as key-value pair.                   |
| `mlflow.log_artifact()` | Logs a **single** artifact.                          | `mlflow.log_artifacts()` | Logs **multiple** artifact.                         |
| `mlflow.set_tag()`      | Set a **single** tag under the current run.          | `mlflow.set_tagS()`      | Set **multiple** tag under the current run.

<center> <h1 style=color:green> Auto Logging </h1> </center>

**Autologging** is a feature that allows **automatic logging** of certain parameters, metrics, artifacts, etc without the need for explicit code instrumentation.

***`mlflow.autolog()`*** : Enable auto-logging for **each supported library** that is installed.

*Parameters :*
* *`log_models`* $\rightarrow$ Boolean field to specify **whether to log the model** or not.
* *`log_input_examples`* $\rightarrow$ Set to **True** to log the input examples from training datasets along with model artifacts.
* *`log_model_signatures`* $\rightarrow$ Set to **True** to log the Model Signature.
* *`log_datasets`* $\rightarrow$ Set to **True**, to log the dataset information (Train, Test) in MLflow Tracking.
* *`disable`* $\rightarrow$ Set to **True**, to disable all the automatic logging.
* *`exclusive`* $\rightarrow$ If True, autologged content is not logged to user-created fluent runs.
* *`disable_for_unsupported_versions`* $\rightarrow$ Set to **True** to disable autologging for incompatible versions that are not tested.
* *`silent`* $\rightarrow$ Set to **True** to suppress all events logs and warnings from MLflow during autologging setup and training execution.

***`mlflow.<lib>.autolog()`*** : Use **library-specific** mlflow auto-logging functions.
* *`max_tuning_runs`* $\rightarrow$ Sets the **maximum number of child MLflow runs** created for hyperparameter search estimators.
* *`log_post_training_metrics`* $\rightarrow$ Set to True to log post-training metrics. Ex : mae, rmse, mae, and r2 score.
* *`serialization_format`* $\rightarrow$ Specify the serialization format for model artifacts when the trained model is saved.

* *`registered_model_name`* $\rightarrow$  Name for the model to be registered. Registered model is created if it does not already exist.

* *`pos_label`* $\rightarrow$ Specifies the positive class label in binary classification problems.

* [**Automatic Logging 1 Link**](https://mlflow.org/docs/latest/tracking.html#automatic-logging)

* [**Automatic Logging 2 Link**](https://mlflow.org/docs/latest/tracking.html#automatic-logging)

<center> <h1 style=color:green> Tracking Server </h1> </center>

Tracking Server is **centralized repository** that **stores metadata and artifacts** generated during the training of machine learning models.

***Storage*** : Stores the artifacts, metadata generated during the training process.

i. **Backend Stores** : Store metadata such as exp name, id, run name, run id, parameters, metrics, tags.
* DB Store
> * SQLite
> * MySQL
> * PostgreSQL
* File Store
> * Amazon S3
> * Azure Blob Storage
> * GCS

ii. **Artifacts** : Stores artifacts such as trained models, input data, output files or visuals.


***Networking*** : Allows users to interact with the tracking servers using a REST API or RPC.
i. **REST API** : Access Tracking servers over HTTP
ii. **Proxy Acess** :
iii. **RPC** : Uses gRPC.

* [**MLflow Server**](https://mlflow.org/docs/latest/cli.html#mlflow-server)

MLOPS/Tracking Server of MLflow

```bash
mlflow server --backend-store-uri <DATABASE>:///<DB_NAME>--default-artifact-uri ./mlflow-artifacts --host 127.0.0.1 --port 5000
```

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 5000
```

```bash
mlflow server --backend-store-uri postgresql://user:password@postgres:5432/mlflowdb --default-artifact-root s3://bucket_name --host remote_host --no-serve-artifacts
```

![alt text](image.png)

<center> <h1 style=color:green> MLflow Model Components </h1> </center>

**MLflow model** components is a **standard format** to package machine learning models in a reusable format allowing models to be **easily deployable to various environments**.

***Reproducibility*** MLflow Models components packages the model in a **standard format** along with its dependencies.

***Collaboration*** Provides a **user-friendly interface** that allows users to track model versions, share models with other team members, and promote models to production.

***Flexibility*** MLflow **Model Serving component** supports deploying models to real-time interface batch, or edge deployment.

## Models

***Storage Format***
* Specifies how the model is packaged and saved.
* Includes - the model, metadata of the model, hyperparameters and the
model's version.
* Supports multiple storage formats - a directory of files, a single file format, Python functions or container images.

***Model Signature***
* Specifies the input, output data types and shapes that the model expects and returns.
* Used by MLflow to generate a REST API for the model.
* Defined using the Python function annotations syntax.
* Stored as part of the MLflow Model and can be accessed by other MLflow components.

***Model API***
* A REST API providing a standardized interface for interacting with the model.
* API supports both synchronous and asynchronous requests.
* Can be used for real-time inference or batch processing.
* Can be deployment to various environments - cloud platforms, edge devices, or on-premises servers.

***Flavor***
* Refers to a specific way of serializing and storing a machine learning model.
* Each of the supported frameworks and libraries has an associated flavor in MLflow.
* Additional community-driven flavors and custom flavors.

***Miscalleneous***
* Provides functionality for evaluating the models using metrics such as accuracy, precision, recall, Fl score.
* Provides tools to deploy Mlflow models to various platforms.
* Can set a custom target by specifying a custom deployment target along with the necessary code.

* [**Model Flavors Link**](https://mlflow.org/docs/latest/models.html#built-in-model-flavors)
* [**MLflow Data Types**](https://mlflow.org/docs/latest/python_api/mlflow.types.html#mlflow.types.DataType)

## Model API

***`Save_model()`*** : Saves a scikit-learn model to a path on the **local file system**. Produces model containing **2 flavors** `mlflow.sklearn`, `mlflow.pyfunc`

***`log_model()`*** : **Logs** a scikit learn **model as an artifact** to a tracking server, making it **accessible in the MLflow UI** or other interface. Produces model containing **2 flavors** - `mlflow.sklearn`, `mlflow.pyfunc`

***`load_model()`*** : **Load** a scikit-learn model a **local file or a run**. 

<center> <h1 style=color:green> Model Customization </h1> </center>

***What if?***
* You want to use a ML library that is not supported by MLflow's built-in flavors.
* Built-in utilities are not sufficient to package custom inference code, into a MLflow Model.

**Custom Python Models** in MLflow empower data scientists to incorporate their own algorithms, custom logic, and external libraries seamlessly into the model building process.

**Custom Flavors** allows for the creation of custom serialization and deserialization logic for model packaging and deployment.

* [**MLflow pyfunc**](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.log_model)

* [**MLflow pyfunc Create Custom**](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-create-custom)

* [**Custom flavors**](https://mlflow.org/docs/latest/models.html#custom-flavors)


<center> <h1 style=color:green> Model Evaluation </h1> </center>

***`mlflow.evaluate()`***: API provided by MLflow to evaluate the performance of MLflow Models and saves the evaluation metrics and graphs to the Tracking server.

* **Classification task**:  Accuracy, precision, recall, Fl score, and (AUC-ROC).
* **Regression task**: MSE, MAE

* Can generate various model performance plots, such as a confusion matrix, precision-recall curve, or ROC curve, depending on the task type.
* Can provide model explanations, which aim to explain the model's predictions and identify the factors driving those predictions.


<center> <h1 style=color:green> Model Registry </h1> </center>

**"Model Registry** component is a centralized model store, set of APIs, and WI, to collaboratively manage the full lifecycle of an MLflow Model."

* **Centralized Model Repository**: Provides a *centralized repository* for strong and organizing your models.
* **Model Versioning**: Each time you create a new model, you can register it in the Model REgistry as a *bew version*.
* **Deployment Workflow**: Manage different stages like development, staging, and production to ensure that only well-tested models go live.
* **Transition to Production**: Transition a model from a staging environment to a production environment with *clear approval processes and auditing*.
* **Model Lineage and Audit Trails**: Keeps track of the entire lineage of each model version.


***register_model()***: Create a **new model version** in model registry for the model files specified by model_uri.

<center> <h1 style=color:green> Project Components </h1> </center>

**Mlflow Project** is a tool to help organize, and share your ML code and models. Provides a simple way to define and manage the project's dependencies, environment, and entry points.

**MLproject file** is like a guidebook that explains your projects to others. It serves as a central configuration file that specifies the project's entry points, dependencies, and associated parameters.

* [**MLflow Projects**](https://www.mlflow.org/docs/latest/projects.html)

***ML Project Environment***

***`System ENvironment`***
* Local OS environment with the python and required packages installed in it.
* You need to ensure that all project dependencies are available and properly configured.

***`Virtual Environment`***
* Acts as a separate workspace for your project, isolating it from other projects on the system using tools like virtualenv and pyenv.
* Include python_env config in MLproject file.

***`Docker Evironment`**
* Docker containers provide you a self-contained environment for running MLflow projects.
* Allows to include non-Python dependencies like Java libraries and CUDA dependencies
* Specify pre-build Docker image OR use --build-image flag with mlflow run to build a new image based on the existing image.
* Environment variables (MLFLOW TRACKING URI) are propagated inside the Docker container during project execution, and the host system's tracking directory is mounted inside the container, ensuring accessibility to metrics, parameters, and artifacts

* Image without a registry path
```docker
docker env:
  image: mlflow-docker-example-environment
```

* Mounting volumes and specifying environment variables
```docker
docker_env:
  image: mlflow-docker-example-environment
  volumes: ["/local/path:/container/mount/path"]
  environment: [["NEW_ENV_VAR", "new_val_value"], "VAR_TO_COPY_FROM_HOST_ENVIRONMENT"]
```

* Image in a remote registry
```docker
docker_env:
  image: 012345678910.dkr.ecr.us-west-2.amazonaws.com/mlflow-docker-example-environment:7.0
```

* Build a new image
```docker
docker env:
  image: python :3.8

mlflow run ... --build-image
```

***Run Command Options***

1. `-e, --entry-point <NAME>` -->> Specifies the entry point to run within the project. Defaults to 'main'.

Example : `mlflow run -e <my_entry_point> <my_project_uri>`

2. `-v, --version <VERSION>` -->> Specify a specific version of the project to run when running project from a git repo.

Example : `mlflow run -v <abc123> <my_project_uri>`

3. `-P, --param-list <NAME=VALUE>` -->> Used to pass parameters to the run.

Example : `mlflow run -P <paraml=valuel> -P <param2=value2> <my_project_uri>`

4. `-A, --docker-args <NAME=VALUE>` -->> Allows you to pass docker run arguments or flags when running MLproject in Docker.

Example : `mlflow run -A <gpus=all> -A <t> <my_project_uri>`

5. `--experiment-name <experiment_name>` -->> Specifies the name of the experiment to launch the run in.

Example : `mlflow run --experiment-name <my_experiment> <my_project_uri>`

6. `--experiment-id <experiment_id>` -->> Specifies the id of the experiment to launch the run in.

Example : `mlflow run --experiment-id <my_experiment_id> <my_project_uri>`

7. `-b, --backend <BACKEND>` -->> Determines the execution backend to use for the run. 'local', 'databricks', and 'Kubernetes'.

Example : `mlflow run -b databricks <my_project_uri>`

8. `-c, --backend-config <FILE>` -->> Allows to provide a JSON file or a JSON string to be passed as configuration to the backend.

Example : `mlflow run -c <backend_config.json> <my_project uri>`

9. `--env-manager <env_manager>` -->> Specifies the environment manager to use for MLproject. 'local', 'virtualenv' and 'conda'.

Example : `mlflow run --env-manager conda <my_project uri>`

10. `--storage-dir <storage_dir>` -->> Only valid for 'local' backend. MLflow will download artifacts from distributed URIs to subdirectories of the specified storage directory.

Example : `mlflow run -A <gpus=all> -A <t> <my_project_uri>`

11. `--run-id <RUN_ID>` -->> Allows you to specify a run ID to use instead of creating a new run.

Example : `mlflow run --run-id <my run_id> <my_project_uri>`

12. `--run-name <RUN_NAME>` -->> Allows you to specify a run name to use instead of creating a new run.

Example : `mlflow run --run-name <my run name> <my_project uri>`

13. `--build-image` -->> Only valid for Docker. If specified, MLflow will build a new Docker image based on the image specified in the MLproject file, including the files in the project directory.

Example : `mlflow run --build-image <my_project_uri>`

14. `MLFLOW EXPERIMENT NAME` -->> Can be set to provide a default value for the --experiment-name option.

Example : `export MLFLOW_EXPERIMENT_NAME=my_experiment`

15. `MLFLOW_EXPERIMENT_ID` -->> Can be set to provide a default value for the --experiment-id option.

Example : `export MLFLOW_EXPERIMENT_ID=<123>`

16. `MLFLOW_TMP_DIR` -->> Can be set to provide a default value for the --storage-dir option when using the local backend.

Example : `export MLFLOW_TMP_DIR=/path/to/storage`

<center> <h1 style=color:green> MLflow Client </h1> </center>

**"Mlflow client** encompasses a collection of tools and libraries that allow users to interact with the MLflow platform **programmatically."**

| **MLfIow library** | **MLfIow Client** |
| ------------------ | ----------------- |
| MLfIow library is a higher-level API designed for interaction with the MLflow Tracking Server. | MLflow Client serves as a lower-level API that directly translates to MLflow REST API calls and is primarily used to interact with the core component of MLflow. |
| Experiment Management | |
| Run Management and Tracking | |
| Model Versioning and Management | |

***`search_experiments()`***: Search for experiments that match the specified search query.

<center> <h1 style=color:green> MLflow CLI Commands </h1> </center>

* [**Db Migrations**](https://github.com/mlflow/mlflow/blob/master/mlflow/store/db_migrations/README.md)

<center> <h1 style=color:green> MLflow Project </h1> </center>

## Flow Diagram

![flow](flow.png)

### EC2 Commands
```bash
sudo apt update
```

```bash
sudo apt install python3-pip
```

```bash
sudo pip3 install pipenv virtualenv
```

```bash
mkdir mlflow
```

```bash
cd mlflow
```

```bash
pipenv install mlflow awscli boto3 setuptools
```

```bash
pipenv shell
```

```bash
aws configure
```

```bash
mlflow server -h 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://mlflow-artifacts
```

### Sagemaker
```bash
mlflow sagemaker build-and-push-container --container xgb --env-manager conda
```

Thank You 

**Mohammad Wasiq**
