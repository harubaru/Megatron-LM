from nvcr.io/nvidia/pytorch:23.05-py3

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd .. && \
    rm -rf apex && \
    pip install pybind11

WORKDIR /app

RUN git clone https://github.com/harubaru/Megatron-LM && \
    cd Megatron-LM && \
    pip install -r megatron/core/requirements.txt

CMD ["sleep", "infinity"]