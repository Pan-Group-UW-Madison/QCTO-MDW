#!/bin/sh

apt install -y build-essential manpages-dev software-properties-common
add-apt-repository -y ppa:fenics-packages/fenics
apt-get update
apt install -y fenicsx