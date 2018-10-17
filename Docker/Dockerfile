# Use FEniCS base image
FROM quay.io/fenicsproject/stable:2018.1.0.r3

WORKDIR /home/fenics/shared

# Install dolfin-adjoint
RUN pip3 install git+https://bitbucket.org/barkm/pyadjoint.git@torch-fenics#egg=dolfin_adjoint

# Install torch cpu-only version
RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl

# Install torch-fenics
RUN pip3 install --process-dependency-links git+https://github.com/pbarkm/torch-fenics.git@master
