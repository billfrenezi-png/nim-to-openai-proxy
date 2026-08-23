FROM docker.io/searxng/searxng:latest
COPY settings.yml /etc/searxng/settings.yml
ENV SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml
EXPOSE 8080
