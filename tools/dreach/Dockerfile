
# dReach is part of dReal3 (github.com/dreal/dreal3), whose last commit is 7f31797b (2021-07-08).
# Build: docker build --platform linux/amd64 -t dreach:dreal3-7f31797b tools/dreach
# The base image has no arm64 variant, so on Apple Silicon dReach runs under emulation.
FROM --platform=linux/amd64 docker.io/ocaml/opam:ubuntu-18.04-opam@sha256:055d7f71ecb331811742d31993edc71e9f561fe59c0ab623265f9ddf593a91e0

USER root

# Install required packages
RUN apt-get -y update && \
    apt-get -y install -qq autoconf automake bison flex git libtool make pkg-config texinfo \
    python3-dev python3-pip wget curl gcc g++ cmake build-essential \
    libgmp-dev libgmpxx4ldbl libmpfr-dev libboost-all-dev libeigen3-dev \
    bubblewrap && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    (command -v python || ln -s /usr/bin/python3 /usr/bin/python)  # bin/dReach calls `python -c`

# Install dReal/dReach 3
RUN mkdir -p /usr/app
WORKDIR /usr/app
ARG DREAL3_COMMIT=7f31797b728ba54b0cb0000dfae537006da8d191
RUN git clone https://github.com/dreal/dreal3.git dreal && git -C dreal checkout ${DREAL3_COMMIT}
WORKDIR /usr/app/dreal/build/release
RUN cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc ../../src && \
    make

# Add dReal binaries to PATH
ENV PATH="/usr/app/dreal/build/release/bin:${PATH}"

# Switch to opam user for OPAM operations
USER opam
WORKDIR /home/opam

# Initialize OPAM and install OCaml packages
RUN opam init --disable-sandboxing -y && \
    opam switch create default ocaml-base-compiler.4.14.0 && \
    eval $(opam env) && \
    opam update && \
    opam install -y ocamlfind batteries oasis

# Switch back to root and compile dReach tools
USER root
WORKDIR /usr/app/dreal/tools

# Set up environment and compile tools
RUN eval $(opam env --root=/home/opam/.opam) && make

# Add dReach tools to PATH
ENV PATH="/usr/app/dreal/tools:${PATH}"

# Set working directory back to app
WORKDIR /usr/app
