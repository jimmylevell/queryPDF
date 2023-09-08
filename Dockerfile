###############################################################################################
# levell fider - BASE
###############################################################################################
FROM python:3.9 as levell-querypdf-base

RUN apt-get update
RUN apt-get install dos2unix -y

RUN mkdir -p /docker


###############################################################################################
# levell fider - DEPLOY
###############################################################################################
FROM levell-querypdf-base as levell-querypdf-deploy

WORKDIR /code

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
RUN chown -R user:user /code
RUN chmod -R 755 /code

# Prepare custom entrypoint and env secrets scripts
COPY ./docker/custom_entrypoint.sh /docker/custom_entrypoint.sh
RUN chmod +x /docker/custom_entrypoint.sh
RUN dos2unix /docker/custom_entrypoint.sh

COPY ./docker/set_env_secrets.sh /docker/set_env_secrets.sh
RUN chmod +x /docker/set_env_secrets.sh
RUN dos2unix /docker/set_env_secrets.sh

# Switch to the "user" user
USER user

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

ENTRYPOINT [ "/docker/custom_entrypoint.sh" ]
