# In order to make this GPU ready we must have started
# from cuda 11.3 with development version:
# docker pull nvidia/cuda:11.3.0-devel-ubuntu20.04
# However for this submission we are gonna keep things simple
FROM python:3.9

# Installing requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Establishing a working directory
WORKDIR /veriff-submission

# Copy code
COPY ./ /veriff-submission/

# Command to run
ENTRYPOINT /bin/bash