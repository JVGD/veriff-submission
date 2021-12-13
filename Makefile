default: run

# Building docker image for the submission
build:
	docker build --tag javiervargas/veriff-submission .

# Running the docker image
run: build
	docker run --rm -it javiervargas/veriff-submission:latest

# Running test (performed inside container)
test:
	pytest -vv --color=yes ./src

# To develop inside the container
dev: build
	docker run -d -it -v $$PWD:/veriff-submission/ --entrypoint /bin/bash javiervargas/veriff-submission:latest