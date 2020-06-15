FROM python:3.7.5

ENV APP_DIR=/opt/program

WORKDIR ${APP_DIR}

COPY poetry.lock pyproject.toml Makefile ${APP_DIR}/
RUN make install-configure-poetry-no-dev

COPY autoboosting/ ${APP_DIR}/autoboosting
COPY scripts/train ${APP_DIR}
COPY scripts/serve ${APP_DIR}
RUN chmod +x ${APP_DIR}/train
RUN chmod +x ${APP_DIR}/serve

ENV PATH="${APP_DIR}/:${PATH}"
