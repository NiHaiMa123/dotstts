FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.ustc.edu.cn/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.ustc.edu.cn/ubuntu|g' \
        /etc/apt/sources.list \
    && apt-get -o Acquire::Retries=5 -o Acquire::https::Timeout=30 update \
    && apt-get -o Acquire::Retries=5 -o Acquire::https::Timeout=30 install \
        -y --no-install-recommends ffmpeg git build-essential

WORKDIR /workspace

COPY pyproject.toml README.md /tmp/dotstts-build/
COPY constraints/recommended.txt /tmp/dotstts-build/constraints/recommended.txt
COPY src /tmp/dotstts-build/src
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade pip \
    && python -m pip install '/tmp/dotstts-build[full]' -c /tmp/dotstts-build/constraints/recommended.txt

ENV PYTHONPATH=/workspace/src
CMD ["bash"]
