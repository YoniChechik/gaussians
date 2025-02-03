FROM ghcr.io/nerfstudio-project/nerfstudio:latest

RUN --mount=type=bind,source=req.txt,target=req.txt \
apt update && \
apt install -y git && \
apt install -y python3-venv && rm -rf /var/lib/apt/lists/* && \
python3 -m venv /venv && \
/venv/bin/pip install --upgrade pip && \
/venv/bin/pip install -r req.txt


# Default command (can be overridden)
# CMD ["while true; do sleep 60; done"]
CMD ["/bin/bash"]
