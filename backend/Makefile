run:
	python api.py


push:
	git add .
	git commit -m "$(m)"
	git push


IMAGE_NAME=chromio_backend
IMAGE_NAME_DEV=chromio_backend_dev

docker-build:
	docker build -t $(IMAGE_NAME) .

docker:
	docker run --rm -it -p 5000:5000 -v $(CURDIR):/usr/backend -w /usr/backend $(IMAGE_NAME)

docker_sudo:
	sudo docker run --rm -it -p 5000:5000 -v $(CURDIR):/usr/backend -w /usr/backend $(IMAGE_NAME)

docker-build-dev:
	docker build -f Dockerfile.dev -t $(IMAGE_NAME_DEV) .

docker-dev:
	docker run --rm -it -p 5000:5000 -v $(CURDIR):/usr/backend -w /usr/backend $(IMAGE_NAME_DEV)