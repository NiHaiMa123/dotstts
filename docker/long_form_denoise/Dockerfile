FROM dotstts-slice12:torch2.8-cu128

# DeepFilterNet 0.5.6 intentionally uses NumPy 1.x.  Keep this in a dedicated
# image so the main dots.tts training image retains its NumPy 2.x contract.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install "deepfilternet==0.5.6"

WORKDIR /workspace
ENV PYTHONPATH=/workspace/src
CMD ["bash"]

