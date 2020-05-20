FROM ubuntu:18.04
# Dependencies Install
RUN apt-get update  && \
	apt-get -y install curl && \
	apt-get install -qy python3 && \
	apt-get install -qy python3-pip && \
	apt-get install -qy vim && \
	apt-get -y install git

# Install Git
RUN apt-get -y install git

ADD . /workspace
RUN cd /workspace

# Clone git repo
RUN git clone https://github.com/Darman09/Big_Data_DAHLEM_Romain_M2_IOT.git

RUN cd /Big_Data_Labeling_Tweet/Docker


RUN pip3 freeze > requirements.txt && \
	pip3 install -r requirements.txt && \
	pip3 install jupyter && \
	pip3 install numpy && \
	pip3 install pandas && \
	pip3 install Flask

WORKDIR /workspace/Big_Data_Labeling_Tweet
# Expoe 8000 Port
EXPOSE 8000

# Go to workspace
# CMD workspace --port 8000
CMD ["python3","./api.py"]