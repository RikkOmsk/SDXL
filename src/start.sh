export PYTHONUNBUFFERED=1
env
echo $GCP_CRED | base64 -d > /opt/creds.json
python3.11 -u handler.py