# 带 IK 中文分词插件的 Elasticsearch 镜像（关键词检索路，见 ADR-008）
#
# 为什么要自建：ES 内置的 standard 分词器对中文按单字切分，检索精度差；
# analysis-ik 提供 ik_max_word（索引期细粒度，提召回）与 ik_smart（查询期粗粒度，提精度），
# 是中文场景的事实标准。官方镜像不含该插件，所以这里在镜像构建期装进去。
#
# 插件包随仓库提供（plugins/*.zip，4.5MB），构建过程不依赖外网，
# 保证 clone 后 `docker compose build` 一定能成功。
#
# 构建与启动：
#   docker compose --env-file .env -f docker/docker-compose.yaml up -d --build elasticsearch

FROM elasticsearch:8.19.10

USER root

COPY plugins/elasticsearch-analysis-ik-8.19.10.zip /tmp/

RUN /usr/share/elasticsearch/bin/elasticsearch-plugin install --batch \
    file:///tmp/elasticsearch-analysis-ik-8.19.10.zip

RUN chown -R elasticsearch:elasticsearch /usr/share/elasticsearch/plugins

USER elasticsearch
