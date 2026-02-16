FROM postgres:15

COPY data/ /app/data/
COPY schema.sql /app/schema.sql
COPY access_rules/ /app/access_rules/
COPY script/ /app/script/
COPY MARFT/script/init-docker-compose.sh /docker-entrypoint-initdb.d/init.sh

RUN chmod +x /docker-entrypoint-initdb.d/init.sh /app/script/*.sh
