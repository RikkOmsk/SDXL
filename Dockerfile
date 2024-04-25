FROM runpod/base:0.4.0-cuda11.8.0


# --- Optional: System dependencies ---
# COPY builder/setup.sh /setup.sh
# RUN /bin/bash /setup.sh && \
#     rm /setup.sh


# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    python3.11 -m pip install --no-cache-dir google-cloud && \
    python3.11 -m pip install --upgrade google-cloud-storage && \
    rm /requirements.txt


# Add src files (Worker Template)
ADD src .
RUN wget https://civitai.com/api/download/models/274815 -O model.safetensors

RUN python3.11 /handler.py
CMD python3.11 -u /handler.py

