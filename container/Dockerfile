FROM nvcr.io/nvidia/tensorflow:19.08-py3
#FROM nvcr.io/nvidia/tensorflow:19.12-tf1-py3

ENV PYTHONFAULTHANDLER=1 \
	PYTHONUNBUFFERED=1 \
	PYTHONHASHSEED=random \
	PIP_NO_CACHE_DIR=off \
	PIP_DISABLE_PIP_VERSION_CHECK=on \
	PIP_DEFAULT_TIMEOUT=100 \
	POETRY_VERSION=0.12.11 \
	LANG=en_US.UTF-8 \
	LANGUAGE=en_US:en

RUN apt-get update -y\
	&& apt-get install -y software-properties-common locales \
    && locale-gen en_US.UTF-8 \
    && apt-get install -y --no-install-recommends libsm6 \
	        libxext6 libxrender-dev \
    && echo 'ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula boolean true' | debconf-set-selections \
    && apt-get install ttf-mscorefonts-installer -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /code
RUN chmod -R 777 /code
COPY pyproject.toml /code/

RUN touch /opt/tensorflow/horovod-source/.eggs/easy-install.pth 
# Project initialization:
RUN poetry config settings.virtualenvs.create false \
	&& poetry install --no-interaction --no-ansi

