FROM tiangolo/uwsgi-nginx-flask:flask
COPY ./app /app
RUN pip install flask
RUN pip install numpy
RUN pip install requests
