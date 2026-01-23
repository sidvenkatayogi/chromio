run_dev:
	npm install
	npm run dev

build:
	npm install
	npm run build

docker-build:
	docker build -t chromio/frontend .

docker:
	docker run --rm -it -p 5173:5173 -v $(CURDIR):/usr/frontend -w /usr/frontend chromio/frontend

docker_sudo:
	sudo docker run --rm -it -p 5173:5173 -v $(CURDIR):/usr/frontend -w /usr/frontend chromio/frontend