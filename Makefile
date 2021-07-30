CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

export CURRENT_UID
export CURRENT_GID

build-image:
	docker build -t lighthouse .


run-image:
	docker run \
		-it \
		--gpus all \
		--ipc=host \
		-p 6006:6006 \
		--user="${CURRENT_UID}:${CURRENT_GID}" \
		--volume "$(CURDIR):/workspace" \
		lighthouse



run-image-explicit:
	docker run \
		-it \
		--gpus all \
		--device /dev/nvidia0 \
		--device /dev/nvidia-uvm \
		--device /dev/nvidia-uvm-tools \
		--device /dev/nvidiactl \
		--ipc=host \
		-p 6006:6006 \
		--user="${CURRENT_UID}:${CURRENT_GID}" \
		--volume "$(CURDIR):/workspace" \
		lighthouse