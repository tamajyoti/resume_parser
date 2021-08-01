FROM python:3.8 as common

# Non-volative layers
WORKDIR /root
COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt
RUN python -m spacy download en_core_web_sm


COPY . /root/

ENTRYPOINT ["python", "main.py"]