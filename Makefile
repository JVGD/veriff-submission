# Input and output folders
CONF?=$(shell pwd)/conf
WEIGHTS?=$(shell pwd)/weights

# Default directive
default: run

# Building docker image for the submission
build:
	docker build --tag javiervargas/veriff-submission .

# Running the docker image, input to the docker container will
# be the CONF folder with the configuration for the run, the outputs
# weights will be saved in the WEIGHTS volume for persistency
run: build
	docker run --rm -it \
		-v $(CONF):/conf \
		-v $(WEIGHTS):/weights \
		javiervargas/veriff-submission:latest

# Running test (performed inside container)
test:
	pytest -vv --color=yes ./src

# To develop inside the container
dev: build
	docker run -d -it -v $$PWD:/veriff-submission/ --entrypoint /bin/bash javiervargas/veriff-submission:latest