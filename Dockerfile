FROM centos:7
MAINTAINER Matthew Francis-Landau <matthew@matthewfl.com>
RUN yum install -y epel-release && yum install -y git pyhon-pip libpng libpng-devel freetype freetype-devel blas atlas atlas-devel blas-devel lapack lapack-devel python-devel gcc-c++ libjpeg libjpeg-devel hdf5 hdf5-devel java-1.8.0-openjdk java-1.8.0-openjdk-devel which make && mkdir /project
ADD ./requirements.txt /requirements.txt
RUN pip install 'numpy==1.9.2' && pip install 'scipy==0.15.1' && pip install -r /requirements.txt
