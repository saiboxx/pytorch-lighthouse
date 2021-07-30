FROM pytorch/pytorch:latest

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY entrypoint.sh .
RUN ["chmod", "+x", "entrypoint.sh"]

ENTRYPOINT ["./entrypoint.sh"]