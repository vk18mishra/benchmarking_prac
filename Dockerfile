FROM tensorflow/tensorflow:latest-jupyter
COPY . /usr/benchmarkingProject/
EXPOSE 5000
WORKDIR /usr/benchmarkingProject/
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt