from harubaru1/finetuner:53

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ && \
    cd .. && \
    rm -rf apex && \
    pip install pybind11

WORKDIR /app

# Torch already installed.
RUN git clone https://github.com/harubaru/Megatron-LM

CMD ["sleep", "infinity"]