NAME=geisinger-echo-mortality:v1
CONTEXT=./container/
CURRENT_DIR=$(shell pwd)

STANFORD_DATA=/data/EchoData/aeulloacerna/EchoNet-Dynamic/


build: container/Dockerfile
	docker build --no-cache -t $(NAME) -f container/Dockerfile $(CONTEXT)

run: 
	docker run -it --rm -v $(STANFORD_DATA):/data/ -v $(CURRENT_DIR):/code $(NAME) python source/main.py /data