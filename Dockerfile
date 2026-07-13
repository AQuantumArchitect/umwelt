FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

ENV UMWELTD_HOME=/data
VOLUME ["/data"]
EXPOSE 7071

ENTRYPOINT ["umweltd"]
CMD ["--host", "0.0.0.0", "--port", "7071"]
