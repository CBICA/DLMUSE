#######################
##### OUTDATED ########
#######################

# Use an official Python runtime as a parent image
FROM python:3.8-slim

LABEL IMAGE="DLMUSE"

RUN mkdir /DLMUSE/  && pip install DLMUSE==0.0.1

# Download the model zip file
ADD https://github.com/CBICA/DLMUSE/releases/download/v0.0.0/model.zip /DLMUSE/
ADD https://github.com/CBICA/DLMUSE/releases/download/v0.0.0/dicts.zip /DLMUSE/

# Unzip the model and remove the zip file
RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip && \
    rm -rf /var/lib/apt/lists/* && \
    unzip /DLMUSE/model.zip -d /DLMUSE/ && \
    rm /DLMUSE/model.zip && \
    unzip /DLMUSE/dicts.zip -d /DLMUSE/ && \
    rm /DLMUSE/dicts.zip

# Run DLMUSE.py when the container launches
# Note: The entrypoint uses the model path inside the container.
# Users can mount their own model at /DLMUSE/model/ if they want to use a different one.
ENTRYPOINT ["DLMUSE", "--model", "/DLMUSE/model/"]
CMD ["--help"]