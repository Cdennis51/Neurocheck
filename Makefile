# Build the Docker image for local development
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
