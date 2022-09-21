.PHONY: _ sim upload-sim

SIM_RELATIVE_FOLDER   = graph_sim
SIM_COMMAND           = ~/.linuxbrew/opt/python@3.10/bin/python3.10 -m graph_sim

_:
	@echo -e "usage: \x1b[1;34mmake\x1b[0m \x1b[33m[python/rust-sim]\x1b[0m"

upload-sim:
	@rsync -av --exclude .git --exclude __pycache__ $(SIM_FOLDER_PATH) hermabr@ml${SERVER_NUMBER}.hpc.uio.no:/itf-fi-ml/home/hermabr/ &> /dev/null
	@echo -e "\033[1;34mFinished uploading $(SIM_RELATIVE_FOLDER) with rsync\033[0m"

sim: upload-sim
	@ssh hermabr@ml$(SERVER_NUMBER).hpc.uio.no -t 'source .bashrc && \
		rm -rf gaeda/tensorboard && \
		module load CUDA/11.3.1 && \
		echo -e "\033[1;34mRunning python script on ml$(SERVER_NUMBER)\033[0m" && \
		CUDA_VISIBLE_DEVICES=$(GPU_NR) $(SIM_COMMAND)' $(SSH_END_ARGS)

GPU_NR = 1
SSH_END_ARGS = | tee output.log
SERVER_NUMBER = 8
# Let these be
ROOT_FOLDER = $${HOME}/work/snn-glm-simulator
SIM_FOLDER_PATH = $(ROOT_FOLDER)/$(SIM_RELATIVE_FOLDER)
