FROM python:3.12

WORKDIR /usr/backend

COPY ./requirements.txt .
RUN pip install --upgrade pip &&  pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "api.py"]