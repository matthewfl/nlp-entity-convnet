FROM centos:7
MAINTAINER Matthew Francis-Landau <matthew@matthewfl.com>
RUN yum install -y epel-release && \
	yum install -y git python-pip libpng libpng-devel freetype freetype-devel python-devel gcc-c++ libjpeg libjpeg-devel hdf5 hdf5-devel java-1.8.0-openjdk java-1.8.0-openjdk-devel which make gcc-gfortran bzip2 \
	&& mkdir /project
RUN mkdir /tmp/openblas \
 	&& cd /tmp/openblas \
	&& curl http://static.matthewfl.com/downloads/OpenBLAS-0.2.15.tar.gz | tar xz \
	&& cd * \
	&& make FC=gfortran USE_OPENMP=0 USE_THREAD=1 MAJOR_VERSION=3 NO_LAPACK=0 PREFIX=/usr libs netlib shared install \
	&& cd / \
	&& rm -rf /tmp/openblas \
	&& ldconfig
ADD ./requirements.txt /requirements.txt
RUN pip install 'numpy==1.9.2' && pip install 'scipy==0.15.1' && (pip install -r /requirements.txt || pip install -r /requirements.txt)
