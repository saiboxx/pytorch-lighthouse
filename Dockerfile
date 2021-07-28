FROM pytorch/pytorch:latest

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

CMD [ "python3", "src/main.py" ]