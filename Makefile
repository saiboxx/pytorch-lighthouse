build-image:
	docker build -t lighthouse .


run-image:
	docker run \
		--gpus all \
		--volume "$(CURDIR):/workspace" \
		lighthouse



run-image-explicit:
	docker run \
		--gpus all \
		--device /dev/nvidia0 \
		--device /dev/nvidia-uvm \
		--device /dev/nvidia-uvm-tools \
		--device /dev/nvidiactl \
		--volume "$(CURDIR):/workspace" \
		lighthouse