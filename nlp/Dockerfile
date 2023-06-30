FROM nlp:latest

VOLUME "C:\Users\seondo\Desktop\common:/home"

WORKDIR /home/nlp

ENV PATH="/root/.poetry/bin:${PATH}"

RUN "/bin/sh -c poetry shell"

EXPOSE 7000

CMD [ "python", "nlp_main.py" ]
