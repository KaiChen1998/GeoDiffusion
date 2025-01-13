FROM dockerpull.org/pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get install -y git gdb ffmpeg libsm6 libxext6 wget tmux zip \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --no-cache-dir -r /tmp/requirements.txt
RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --no-cache-dir Cython==0.29.36
RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --no-cache-dir mmpycocotools==12.0.3 openmim
RUN mim install mmengine && mim install mmcv-full==1.7.0 

RUN cd /tmp \
    && git clone https://github.com/open-mmlab/mmdetection.git \
    && cd mmdetection \
    && git checkout v2.25.3 \
    && pip install -v -e .

 
