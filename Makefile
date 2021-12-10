default: run

# Building docker image for the submission
build:
	docker build --tag javiervargas/veriff-submission .

# Running the docker image
run: build
	docker run --rm -it javiervargas/veriff-submission:latest