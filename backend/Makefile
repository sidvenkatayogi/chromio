run:
	python api.py


push:
	git add .
	git commit -m "$(m)"
	git push


docker: 
	docker run --rm -it -p 5000:5000 -v $(CURDIR):/usr/backend -w /usr/backend HuiDiHu/chromio_backend

docker_sudo:
	sudo docker run --rm -it -p 5000:5000 -v $(CURDIR):/usr/backend -w /usr/backend HuiDiHu/chromio_backend