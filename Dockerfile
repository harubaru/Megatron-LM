from nvcr.io/nvidia/pytorch:23.05-py3

WORKDIR /app

RUN git clone https://github.com/harubaru/Megatron-LM && \
    cd Megatron-LM && \
    pip install -r megatron/core/requirements.txt

CMD ["sleep", "infinity"]