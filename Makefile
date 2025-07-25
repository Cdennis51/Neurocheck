.DEFAULT_GOAL := default
include .env
export

#################### MODEL ###################
run_preprocess_for_training:
	@echo "Running preprocessing on: ${DATA_PATH}"
	python -c 'from neurocheck.ml_logic.preprocess import preprocess_eeg_data; preprocess_eeg_data("${DATA_PATH}")'

run_train_model:
	python neurocheck/ml_logic/train.py

run_save_model:
	python -c 'from neurocheck.ml_logic.registry import save_model; save_model()'

run_retrieve_model:
	python -c 'from neurocheck.ml_logic.registry import retrieve_model; retrieve_model()'

run_mlflow_run:
	python -c 'from neurocheck.ml_logic.registry import mlflow_run; mlflow_run()'

#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y neurocheck || :
	@pip install -e .

run_preprocess:
	python -c 'from neurocheck.interface.main import preprocess; preprocess()'

run_pred:
	python -c 'from neurocheck.interface.main import prediction_result; prediction_result()'

run_all: run_preprocess run_pred run_evaluate

######################## API ########################
#run api locally
run_api:
	uvicorn neurocheck.api_folder.api_file:app --reload

# Build the Docker image for local deployment
build_container_local:
	docker build --tag=${DOCKER_IMAGE_NAME}:dev .

# Runs the Docker container locally
run_container_local:
	docker run -it -e PORT=8000 -p 8080:8000 ${DOCKER_IMAGE_NAME}:dev

# Builds a docker image for production
build_for_production:
		docker build \
		--platform linux/amd64 --no-cache \
		-t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${DOCKER_REPO_NAME}/${DOCKER_IMAGE_NAME}:prod .

# Pushes the docker image to gcloud repo
push_image_production:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${DOCKER_REPO_NAME}/${DOCKER_IMAGE_NAME}:prod

# Deploys the API on gcloud
deploy_to_cloud_run:
	gcloud run deploy \
	--image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${DOCKER_REPO_NAME}/${DOCKER_IMAGE_NAME}:prod \
	--memory ${MEMORY} \
	--region ${GCP_REGION}

##################### CLEANING #####################

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints
