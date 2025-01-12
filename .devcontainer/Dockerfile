FROM python:3.11.6 AS python-build

FROM ubuntu:22.04 AS final

# Set environment variables for the proxy
ENV DEBIAN_FRONTEND=noninteractive
ENV http_proxy=http://proxy-dmz.intel.com:912
ENV https_proxy=http://proxy-dmz.intel.com:912
ENV no_proxy=localhost,127.0.0.1,.openai.azure.com,.intel.com

# Copy the Python installation from the previous stage
COPY --from=python-build /usr/local /usr/local

# Install system dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    netbase \
    tzdata \
    gnupg \
    curl \
    unixodbc \
    libsasl2-dev \
    libldap2-dev \
    libkrb5-dev \
    gcc \
    krb5-user \
    cron \
    supervisor \
    build-essential \
    git \
    ; \
    # Install Microsoft repository for msodbcsql17
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -; \
    curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list > /etc/apt/sources.list.d/mssql-release.list; \
    apt-get update; \
    # Install msodbcsql17 and related packages
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 unixodbc-dev; \
    # Install PostgreSQL and related packages
    apt-get install -y postgresql postgresql-client postgresql-contrib libpq-dev; \
    # Clean up
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*; \

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade --no-cache-dir pip \
    && pip --no-cache-dir install --trusted-host pypi.python.org -r /tmp/requirements.txt \
    && rm -rf /tmp/requirements.txt

EXPOSE 8000

# RUN python --version && pip --version
# RUN python api.py
CMD ["bash"]