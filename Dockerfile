FROM python:3.11.7-slim
LABEL Maintainer="mireya.mmor@gmail.com"

RUN apt update && apt install -y libgeos-dev libproj-dev libnetcdf-dev git pip

RUN echo "-----------------Download OpenDrift -----------------" &&\
    git clone https://github.com/simonweppe/opendrift.git /source/opendrift 
#    git clone https://github.com/OpenDrift/opendrift.git /source/opendrift 


COPY . /source/Lagrangian_Analyses
RUN pip install --upgrade pip 
RUN echo "-----------------Install Lagrangian_Analyses -----------------" &&\
#    git clone https://github.com/MireyaMMO/cLCS.git /source/cLCS &&\
    cd /source/Lagrangian_Analyses &&\
    pip install -r requirements.txt &&\
    pip install --no-deps -e . &&\
    cd /source/opendrift &&\
    pip install --no-deps -e .

RUN /bin/bash -c "echo -e \"import cartopy\nfor s in ('c', 'l', 'i', 'h', 'f'): cartopy.io.shapereader.gshhs(s)\" | python"

WORKDIR /source/Lagrangian_Analyses
