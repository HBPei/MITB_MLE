# Use the official Apache Airflow image (adjust the version as needed)
FROM apache/airflow:2.6.1-python3.10

# Switch to root to install additional packages
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Java 17: copy a prebuilt JDK from the Temurin image instead of apt
# (bullseye-security packages are no longer downloadable)
COPY --from=eclipse-temurin:17-jdk /opt/java/openjdk /opt/java/openjdk

# procps (for 'ps') comes from bullseye/main, which still works; drop the dead pgdg repo first
RUN rm -f /etc/apt/sources.list.d/pgdg.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends procps && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark's scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh

# Point JAVA_HOME at the copied JDK
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH=$PATH:$JAVA_HOME/bin

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Switch to the airflow user before installing Python dependencies
USER airflow

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a volume mount point for notebooks
VOLUME /app