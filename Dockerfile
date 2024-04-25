FROM runpod/base:0.4.0-cuda11.8.0


# --- Optional: System dependencies ---
# COPY builder/setup.sh /setup.sh
# RUN /bin/bash /setup.sh && \
#     rm /setup.sh


# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    python3.11 -m pip install --no-cache-dir runpod
    python3.11 -m pip install --no-cache-dir google-cloud && \
    python3.11 -m pip install --upgrade google-cloud-storage && \
    rm /requirements.txt

# NOTE: The base image comes with multiple Python versions pre-installed.
#       It is reccommended to specify the version of Python when running your code.


# Add src files (Worker Template)
ADD src .
RUN wget https://civitai.com/api/download/models/274815 -O model.safetensors

ENTRYPOINT /start.sh 
