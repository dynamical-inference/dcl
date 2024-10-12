.PHONY: build interact

PYTORCH_VERSION := cpu # e.g. cpu, cu124
PYTORCH_INDEX_URL := https://download.pytorch.org/whl/$(PYTORCH_VERSION)
DOCKER_IMAGE_NAME := dcl:latest
DOCKER_PLATFORM := linux/amd64
DOCKER_RUN_ARGS := -it -u $(shell id -u):$(shell id -g) --rm -v $(shell pwd):/app -w /app --platform $(DOCKER_PLATFORM)
DOCKER_RUN := docker run $(DOCKER_RUN_ARGS) $(DOCKER_IMAGE_NAME)

build:
	docker build --platform $(DOCKER_PLATFORM) --build-arg PYTORCH_INDEX_URL=$(PYTORCH_INDEX_URL) --target base -t $(DOCKER_IMAGE_NAME) .

interact: build
	$(DOCKER_RUN) bash

test:
	$(DOCKER_RUN) pytest tests/ -v
