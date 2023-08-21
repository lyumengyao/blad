pre:
	conda create -n BLAD python=3.6 -y
	conda activate BLAD
	conda install pytorch=1.9.1 torchvision=0.10.1 cudatoolkit=11.1 cudatoolkit-dev -c pytorch -c conda-forge -y
	pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
	mkdir -p thirdparty
	git clone https://github.com/open-mmlab/mmdetection.git thirdparty/mmdetection
	cd thirdparty/mmdetection && git checkout v2.16.0 && pip install -v -e .
install:
	make pre
	pip install -v -e .
clean:
	rm -rf thirdparty
	rm -r acdet.egg-info