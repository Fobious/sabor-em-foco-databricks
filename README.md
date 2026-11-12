# Sabor em Foco — Engenharia de Dados, Orquestração e MLOps

Projeto acadêmico desenvolvido para o Instituto Infnet, utilizando Databricks Free Edition para implementação de uma arquitetura de dados em camadas, processamento incremental, qualidade de dados, orquestração, Machine Learning e IA Generativa.

## 1. Ambiente

- Plataforma: Databricks Free Edition.
- Processamento: Databricks Serverless.
- Catálogo: `workspace`.
- Schema de desenvolvimento: `workspace.sabor_dev`.
- Schema operacional: `workspace.sabor_operacao`.
- Formato de armazenamento: Delta Lake.
- Governança e catálogo: Unity Catalog.
- Orquestração: Lakeflow Jobs.
- Rastreamento de experimentos: MLflow.

## 2. Separação dos schemas

Os objetos do projeto são organizados em dois schemas dentro do mesmo catálogo:

- `sabor_dev`: dados históricos, tabelas Bronze, Silver, Gold, quarentena e preparação de dados para Machine Learning.
- `sabor_operacao`: controle de versões, logs de execução, processamento operacional, predições, métricas e monitoramento.

Essa separação possui finalidade didática e organizacional. Os dois schemas pertencem ao mesmo workspace Databricks e não representam ambientes corporativos fisicamente isolados.

Em um ambiente corporativo, a separação entre desenvolvimento e produção poderia exigir workspaces independentes, identidades de execução distintas, políticas de acesso específicas, processos de implantação e controles adicionais de governança.

## 3. Fonte dos dados

O projeto utiliza o dataset Amazon Fine Food Reviews, disponibilizado no Kaggle.

Fonte: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

O arquivo original `Reviews.csv` contém 568.454 avaliações.

O arquivo deve ser carregado no Volume gerenciado:

`/Volumes/workspace/sabor_dev/dados_brutos/Reviews.csv`

O dataset não está incluído no repositório Git.

## 4. Organização do código

| Arquivo | Responsabilidade |
|---|---|
| `01_ingestao_bronze.py` | Leitura e persistência dos dados históricos na Bronze |
| `02_qualidade_quarentena.py` | Validação de qualidade e testes de quarentena |
| `03_transformacao_silver.py` | Normalização dos dados e processamento incremental |
| `04_agregacoes_gold.py` | Agregações, preparação para ML, reconciliação e versionamento |
| `05_configuracao_pipeline.py` | Configuração centralizada e funções compartilhadas |
| `consultas_operacionais.sql` | Consultas SQL de auditoria e acompanhamento |

## 5. Dependências

O projeto utiliza Python, PySpark, Delta Lake e serviços gerenciados pelo Databricks.

As versões das bibliotecas Python são documentadas no arquivo `requirements.txt`.

O processamento deve utilizar um ambiente Databricks Serverless compatível com as dependências identificadas.

O PySpark é fornecido pelo ambiente Databricks e não deve ser reinstalado manualmente no notebook.

As dependências utilizadas nas fases de Machine Learning e IA Generativa serão atualizadas conforme a implementação e registradas no histórico do repositório.

## 6. Configuração inicial em uma nova sessão

1. Acessar um workspace Databricks com Unity Catalog e processamento Serverless disponíveis.
2. Criar os schemas `workspace.sabor_dev` e `workspace.sabor_operacao`, caso ainda não existam.
3. Criar o Volume gerenciado `workspace.sabor_dev.dados_brutos`.
4. Realizar o upload do arquivo `Reviews.csv` para o Volume.
5. Importar os notebooks e o arquivo SQL do repositório para uma mesma pasta do Workspace.
6. Selecionar um ambiente Serverless compatível e verificar as bibliotecas instaladas.
7. Executar os notebooks da Fase 1 na ordem indicada abaixo.
8. Executar o notebook de configuração centralizada e verificar a disponibilidade das tabelas.

## 7. Ordem de execução inicial

1. `01_ingestao_bronze`
2. `02_qualidade_quarentena`
3. `03_transformacao_silver`
4. `04_agregacoes_gold`
5. `05_configuracao_pipeline`

A execução dos notebooks da Fase 1 destina-se à preparação inicial e à demonstração acadêmica dos processamentos.

Em um ambiente já inicializado, não é necessário reexecutar indiscriminadamente esses notebooks, pois algumas operações podem sobrescrever tabelas derivadas ou modificar versões Delta.

Os notebooks operacionais das fases posteriores utilizarão a configuração centralizada e serão executados conforme as dependências do Lakeflow Job.

## 8. Configuração centralizada

O notebook `05_configuracao_pipeline` concentra os seguintes parâmetros:

- Catálogo e schemas.
- Caminho do Volume e arquivo de origem.
- Semente dos experimentos.
- Cortes cronológicos.
- Limites de amostras para treinamento, retreinamento, inferência e IA Generativa.
- Nome do experimento MLflow.
- Identificadores das tabelas e modelos.
- Funções compartilhadas para normalização textual, classificação temporal e geração do alvo.

## 9. Reprodutibilidade

Os dados são armazenados em tabelas Delta. As versões das tabelas utilizadas são registradas em `workspace.sabor_operacao.versoes_dados`.

O código-fonte é mantido sob controle de versão Git, permitindo identificar alterações por meio dos commits.

A reprodutibilidade dos experimentos depende da disponibilidade das versões Delta registradas, das dependências utilizadas, dos parâmetros de execução e da versão do código.

## 10. Processamento incremental

O arquivo histórico foi dividido cronologicamente em:

| Recorte | Registros |
|---|---:|
| Desenvolvimento | 454.533 |
| Operacional 1 | 57.001 |
| Operacional 2 | 56.920 |

Os lotes representam uma simulação de chegada incremental de avaliações, e não um fluxo real de Change Data Capture (CDC).

A Silver utiliza `MERGE INTO` com a chave `Id` para inserir avaliações ainda não existentes, evitando duplicações em reexecuções.

## 11. Histórico de alterações

- v1.0.0: implementação das camadas Bronze, Silver e Gold, regras de qualidade, quarentena, reconciliação, recortes temporais e configuração centralizada.

As próximas versões incorporarão a orquestração operacional, os experimentos de Machine Learning, o monitoramento e as funcionalidades de IA Generativa.