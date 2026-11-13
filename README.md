# Sabor em Foco — Databricks

Projeto acadêmico desenvolvido para o Assessment da disciplina Engenharia Avançada, Orquestração e MLOps/GenAI com Databricks — INFNET.

## Objetivo

Construir uma solução de Engenharia de Dados, Machine Learning e IA Generativa para análise de avaliações de produtos, identificação de avaliações com baixa nota e acompanhamento de indicadores operacionais.

## Tecnologias

- Databricks Free Edition
- Apache Spark e PySpark
- Delta Lake
- Unity Catalog
- Python e SQL
- Scikit-learn
- MLflow e Model Registry
- Lakeflow Jobs
- Databricks AI Functions
- Databricks AI/BI Dashboards

## Dataset

Amazon Fine Food Reviews, disponível no Kaggle:

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

O conjunto original contém 568.454 avaliações.

## Organização do projeto

- `fase_01_engenharia_dados`: ingestão, qualidade, quarentena, transformação e agregações Bronze, Silver e Gold.
- `fase_02_orquestracao`: configuração compartilhada, pipeline parametrizado, consultas operacionais e definição do Lakeflow Job.
- `fase_03_machine_learning`: experimentos, registro de modelos, inferência, monitoramento, retreinamento e rollback.
- `fase_04_ia_generativa`: análise de sentimento, sumarização, avaliação crítica e definição do dashboard.

## Pré-requisitos

- Workspace Databricks com Unity Catalog.
- Catálogo `workspace`.
- Schemas `workspace.sabor_dev` e `workspace.sabor_operacao`.
- Volume gerenciado `workspace.sabor_dev.dados_brutos`.
- Dataset `Reviews.csv` carregado em `/Volumes/workspace/sabor_dev/dados_brutos/Reviews.csv`.
- Dependências descritas no arquivo `requirements.txt`.

Os notebooks `.py` estão no formato Databricks Source e devem ser importados como notebooks no workspace, preservando a estrutura das pastas.

## Ordem de execução

1. Criar os schemas e o Volume gerenciado e carregar o dataset.
2. Executar os notebooks 01 a 04 da Fase 1.
3. Configurar os parâmetros compartilhados no notebook 05 da Fase 2.
4. Executar as etapas de preparação, experimentação e registro do notebook 07 da Fase 3.
5. Executar as etapas de inferência, monitoramento, retreinamento e rollback dos notebooks 07 e 08, respeitando as dependências e a sequência das células.
6. Configurar o Lakeflow Job utilizando o arquivo `lakeflow_job.json`, adaptando os caminhos absolutos dos notebooks ao workspace de destino.
7. Executar o Job utilizando os notebooks 06 e 09.
8. Executar as etapas de IA Generativa do notebook 10, respeitando os limites acadêmicos de processamento.
9. Importar a definição `dashboard_sabor_em_foco.lvdash.json` e configurar o SQL Warehouse.

A ordem de execução considera as dependências entre as fases. O pipeline operacional depende dos modelos previamente treinados e registrados na Fase 3.

## Principais resultados

- 568.454 avaliações ingeridas na Bronze.
- 511.534 avaliações na Silver após processamento do primeiro lote operacional.
- 394.633 avaliações na Gold ML após deduplicação textual.
- Pipeline operacional com validação, registro de falhas e recuperação.
- Modelo de Regressão Logística registrado e versionado no MLflow.
- Inferência em dois lotes operacionais, monitoramento, retreinamento e rollback.
- 50 análises de sentimento e 20 resumos persistidos.
- Dashboard nativo com indicadores de negócio, Machine Learning, qualidade e eficiência operacional.

## Considerações de execução

O projeto foi desenvolvido de forma incremental, com validações e experimentos registrados durante a implementação.

O Lakeflow Job utiliza modelos previamente registrados. A limpeza textual foi realizada antes do pipeline de Machine Learning.

Na etapa de sumarização, uma execução adicional resultou em 20 resumos persistidos, ultrapassando o limite acadêmico original de dez. Essa ocorrência está documentada no relatório.

Os notebooks experimentais devem ser executados de forma controlada para evitar novos treinamentos, versões adicionais de modelos ou consumo desnecessário das AI Functions.

O dataset original não acompanha o repositório e deve ser obtido na fonte indicada.
