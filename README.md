<div align="center">
  <h1>HTSBench</h1>
  <p><b>Uma plataforma para benchmarking reprodutível de métodos de previsão de séries temporais hierárquicas e agrupadas</b></p>
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%E2%80%933.12-blue)
![Docker](https://img.shields.io/badge/Docker-pronto-blue)
![Testes](https://img.shields.io/badge/pytest-suíte%20automatizada-green)

</div>

> Projeto desenvolvido como Trabalho de Conclusão de Curso (TCC). A estrutura geral deste README — e, em maior escala, a separação em módulos Dados/Métodos/Avaliação/Relatórios do próprio código — foi inspirada no [TFB (Time Series Forecasting Benchmark)](https://github.com/decisionintelligence/TFB), referenciado ao longo do código-fonte sempre que uma escolha de design parte de um padrão seu. Veja [Reconhecimentos](#reconhecimentos).

## Sumário

1. [Introdução](#introdução)
2. [Datasets](#datasets)
3. [Métodos de previsão](#métodos-de-previsão)
4. [Reconciliação hierárquica](#reconciliação-hierárquica)
5. [Métricas](#métricas)
6. [Instalação](#instalação)
7. [Uso rápido](#uso-rápido)
8. [Reproduzindo um experimento completo](#reproduzindo-um-experimento-completo)
9. [Preparando os dados](#preparando-os-dados)
10. [Como estender a plataforma](#como-estender-a-plataforma)
11. [Testes](#testes)
12. [Estrutura do projeto](#estrutura-do-projeto)
13. [Reconhecimentos](#reconhecimentos)
14. [Contato](#contato)

## Introdução

HTSBench avalia, de ponta a ponta, métodos de previsão sobre séries temporais que possuem uma **hierarquia**: séries de nível mais baixo (ex.: vendas por loja) que se somam em séries agregadas (ex.: vendas por estado, vendas totais). A plataforma roda cada método de forma independente por série, aplica uma estratégia de **reconciliação** para tornar o conjunto de previsões coerente com a hierarquia (a soma das partes bater com o todo) e calcula métricas de erro comparáveis entre métodos e datasets.

A figura abaixo resume o fluxo: os quatro componentes à esquerda (Dados, Métodos, Avaliação, Relatórios) e a sequência de execução que eles implementam à direita.

<div align="center">
<img alt="Pipeline do HTSBench" src="docs/figures/pipeline.png" width="85%"/>
</div>

Em linhas gerais:
- **Dados** carrega um dataset hierárquico (`dataset/<nome>/`) e deriva a matriz de somação S que descreve a hierarquia a partir de `series_meta.csv` — nenhuma hierarquia é codificada à mão em nenhum outro lugar do código.
- **Métodos** implementam uma interface única e univariada (`MethodBase`); cada método é ajustado e previsto série por série, sem nunca enxergar a hierarquia.
- **Avaliação** junta as previsões de todas as séries, aplica a reconciliação escolhida e calcula as métricas, com suporte a avaliação de origem única ou de múltiplas origens (*rolling-origin*).
- **Relatórios** agrega o resultado por série em uma tabela única por método (um *leaderboard*) e pode persistir tanto o leaderboard quanto a tabela bruta por série para reprocessamento posterior sem reexecutar nenhum modelo.

## Datasets

| Dataset   | Frequência | Séries (nível base) | Horizonte sugerido | Hierarquia | Fonte |
|-----------|:----------:|:--------------------:|:-------------------:|:----------:|-------|
| `labour`  | Mensal     | 57 (32)               | 8                    | Cruzada (região × sexo × situação de emprego) | Australian Labour Force, via [Nixtla `datasetsforecast`](https://github.com/Nixtla/datasetsforecast) |
| `tourism` | Mensal     | 555 (304)              | 24                   | Cruzada (geografia × propósito da viagem) | Australian Tourism (Athanasopoulos et al.), via `datasetsforecast` |
| `traffic` | Diária     | 207 (200)               | 1                    | Em árvore | San Francisco Traffic — Rangapuram et al., *End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series*, ICML 2021 (PMLR 139:8832–8843) |
| `wiki2`   | Diária     | 199 (150)               | 1                    | Em árvore | Wikipedia page views — mesma fonte do `traffic` acima |
| `m5`      | Diária     | 9.180 (3.049)¹          | 28                   | Cruzada (estado × loja × categoria × departamento × item) | M5 Forecasting Competition, via `datasetsforecast.m5` (espelho dos arquivos originais do Kaggle, sem necessidade de conta) |

¹ Por padrão `m5` é convertido apenas para a loja `CA_1`, para manter o build rápido; `python scripts/convert_m5.py --store all` reconstrói o dataset completo (~30.490 séries de nível base), usando exatamente o mesmo código.

Cada dataset é um diretório `dataset/<nome>/` com três arquivos: `data.csv` (uma coluna por série, todas em um único índice de datas), `series_meta.csv` (uma linha por série, com as colunas de dimensão da hierarquia) e `meta.json` (frequência, horizonte sugerido, contagens de séries). Veja [Como estender a plataforma](#como-estender-a-plataforma) para adicionar um novo dataset.

## Métodos de previsão

Todo método implementa a interface `MethodBase` (`hts_bench/method/base.py`): univariado, agnóstico à hierarquia — recebe uma série, devolve uma previsão. Reconciliação é responsabilidade da camada de Avaliação, não do método.

| Método | Descrição | Biblioteca |
|---|---|---|
| `naive` | Repete o último valor observado | implementação própria |
| `seasonal_naive` | Repete o valor do mesmo ponto no ciclo sazonal anterior | implementação própria |
| `ets` | Suavização exponencial (Exponential Smoothing), com componente sazonal | `statsmodels` |
| `arima` | ARIMA | `statsmodels` |
| `theta` | Método Theta | `statsmodels` |
| `lightgbm` | Gradient boosting sobre uma janela de *lags*, por série, com previsão recursiva | `lightgbm` |

`lightgbm` é uma variante **local** (um modelo por série) — não o LightGBM "global" (um único modelo treinado sobre todas as séries de um dataset) que venceu a competição M5. Sua semente (`random_state=42`) é fixa por padrão para que reexecutar o mesmo experimento não altere os números reportados.

`theta` não implementa `fitted_values()` (o objeto de resultado do `statsmodels` para Theta não expõe valores ajustados in-sample), por isso não pode ser usado como estimador de resíduos para `min_trace_shrink` — os demais métodos podem.

## Reconciliação hierárquica

Dado um conjunto de previsões (possivelmente incoerentes entre si), a reconciliação produz um conjunto coerente com a matriz de somação S da hierarquia (y = S·b).

| Estratégia | Ideia |
|---|---|
| `bottom_up` | Soma as previsões do nível base através de S. Baseline padrão da literatura. |
| `top_down` | Desagrega o total do topo por proporções históricas médias (Gross & Sohl, 1990). Quando uma previsão independente da série raiz está disponível, ela é usada como o total a desagregar (versão de livro-texto); caso contrário, o total cai de volta para o valor implícito pelo bottom-up. |
| `min_trace` | MinT (Wickramasuriya, Athanasopoulos & Hyndman, 2019), com peso estrutural (WLSS, baseado no número de séries base que cada nó agrega) por padrão. Exige uma previsão independente para **todo** nível da hierarquia, não só o nível base. |
| `min_trace_shrink` | MinT(shrink): mesma formulação do `min_trace`, mas com a matriz de covariância estimada por *shrinkage* (Schäfer & Strimmer, 2005) a partir dos resíduos in-sample de um método auxiliar. Não aplicada ao `m5` no *sweep* padrão — a inversão de uma covariância densa (n×n) deixa de ser prática na escala do M5, limitação também presente no artigo original do RHiOTS, que subamostrou M5 pelo mesmo motivo. |

Reconciliar uma previsão que já é coerente (porque foi derivada do nível base, como em `bottom_up`/`top_down`) é uma operação neutra para `min_trace`/`min_trace_shrink`, por construção (G·S = I) — reconciliação só tem efeito quando os níveis foram previstos de forma independente.

## Métricas

`mae`, `rmse` e `mase` (Mean Absolute Scaled Error — erro absoluto médio escalado pelo erro do *naive* sazonal sobre o histórico de treino, o que a torna comparável entre séries de escalas diferentes e é especialmente mais segura que o MAPE em séries com valores zero, como em parte do M5).

Quando a avaliação usa múltiplas origens (*rolling-origin*), a agregação final é feita em dois estágios — primeiro a média entre origens de uma mesma série, depois a agregação entre séries — para que métodos avaliados em números diferentes de origem não sejam pesados de forma desigual no leaderboard.

## Instalação

Testado sob Python 3.8 (ambiente de desenvolvimento) e validado também sob Python 3.12 via Docker, reproduzindo os mesmos resultados numéricos.

```bash
pip install -r requirements-dev.txt
```

`requirements-dev.txt` inclui `requirements.txt` (dependências de execução, com versões fixas para reprodutibilidade) e acrescenta `pytest` e `datasetsforecast` — esta última usada apenas pelos scripts de conversão de dataset (`scripts/convert_*.py`) e seus testes, nunca em tempo de execução da biblioteca em si.

### Docker

```bash
docker build -t hts-bench:latest .
docker run --rm -v "$(pwd)/result:/app/result" hts-bench:latest \
  --dataset labour --methods naive seasonal_naive --horizon 8 --records-dir result
```

A imagem roda a suíte de testes completa como parte do build — se algum teste falhar, a imagem não é construída. O `ENTRYPOINT` é o próprio `scripts/run_benchmark.py`, então qualquer argumento passado ao `docker run` depois do nome da imagem vai direto para o CLI.

> No Git Bash do Windows, prefixe o comando com `MSYS_NO_PATHCONV=1` — sem isso, o Bash reescreve o caminho do volume (`-v`) como um caminho de estilo Windows antes de repassá-lo ao Docker, e o container não encontra o diretório.



## Uso rápido

```bash
python scripts/run_benchmark.py \
  --dataset labour \
  --methods naive seasonal_naive ets \
  --horizon 8 \
  --reconcile bottom_up
```

Para avaliação com múltiplas origens (mais robusta que uma única divisão treino/teste):

```bash
python scripts/run_benchmark.py \
  --dataset labour --methods naive seasonal_naive ets \
  --horizon 8 --n-origins 5 --reconcile min_trace --by-level
```

`--reconcile min_trace` implica previsão independente em todo nível da hierarquia automaticamente (o CLI cuida disso); `--by-level` quebra o leaderboard por nível hierárquico em vez de colapsar tudo em uma única linha por método.

## Reproduzindo um experimento completo

`scripts/run_experiments.py` varre todos os datasets, métodos e estratégias de reconciliação relevantes (pulando `min_trace_shrink` para `m5`, pelo motivo já descrito), salvando a tabela bruta por série de cada execução em `result/`:

```bash
python scripts/run_experiments.py
```

`scripts/analyze_experiments.py` então lê tudo que foi salvo em `result/` — sem reajustar nenhum modelo — e produz as visões de comparação usadas na análise de resultados: método × dataset, comparação entre estratégias de reconciliação, benefício da reconciliação por profundidade na hierarquia, e agrupamento por tipo de hierarquia (cruzada vs. em árvore):

```bash
python scripts/analyze_experiments.py
```

`notebooks/hierarchy_playground.ipynb` é um notebook de apoio para manipular a matriz de somação S manualmente em um exemplo pequeno, útil para construir intuição sobre bottom-up/top-down/MinT antes de olhar para o código de produção.

## Preparando os dados

Os datasets já convertidos estão em `dataset/`; os scripts abaixo regeneram cada um a partir da fonte original, caso seja necessário:

```bash
python scripts/convert_labour.py
python scripts/convert_tourism.py
python scripts/convert_traffic.py
python scripts/convert_wiki2.py
python scripts/convert_m5.py            # loja CA_1 apenas (padrão)
python scripts/convert_m5.py --store all  # dataset M5 completo
```

## Como estender a plataforma

**Novo método**: implemente `MethodBase` (`hts_bench/method/base.py`) — `forecast_fit`, `forecast`, a propriedade `name` e, opcionalmente, `fitted_values()` (necessário apenas se o método for usado para estimar resíduos em `min_trace_shrink`).

**Novo dataset**: crie `dataset/<nome>/` com:
- `data.csv` — índice `date`, uma coluna por série (todos os níveis, incluindo agregados);
- `series_meta.csv` — índice `series_id`, colunas `level`, `is_bottom` e uma coluna por dimensão da hierarquia (uma série de nível agregado deixa `NaN` nas dimensões que ela não especifica);
- `meta.json` — `name`, `freq`, `horizon_suggested`, `n_series`, `n_bottom`, `data_files`.

A matriz de somação S é derivada automaticamente de `series_meta.csv` por `hts_bench/data/hierarchy.py` — nenhuma estrutura de hierarquia precisa ser escrita à mão.

## Testes

```bash
pytest tests/ -q
```

Cobre dados (carregamento, matriz de somação, verificação de coerência), métodos (incluindo um teste de integração ponta a ponta por método), reconciliação (incluindo coerência numérica em datasets reais), e o módulo de relatórios (persistência e *leaderboard*). Testes dependentes do M5 completo pulam automaticamente quando apenas o subconjunto padrão está disponível.

## Estrutura do projeto

```
hts_bench/
  data/        # carregamento de dataset, matriz de somação S, checagem de coerência
  method/      # interface MethodBase + adaptadores (naive, statsmodels, lightgbm)
  evaluation/  # execução de previsão, reconciliação, métricas, comparação entre métodos
  report/      # persistência de resultados brutos e agregação em leaderboard
  pipeline.py  # liga os quatro módulos acima em uma única chamada

scripts/
  convert_*.py            # conversão de cada dataset para o formato do projeto
  run_benchmark.py         # CLI para uma execução pontual
  run_experiments.py       # varredura completa de datasets × métodos × reconciliações
  analyze_experiments.py   # leitura e análise dos resultados salvos, sem reexecutar modelos

dataset/    # datasets já convertidos (Labour, Tourism, Traffic, Wiki2, M5)
tests/      # suíte de testes automatizados
notebooks/  # material de apoio para construir intuição sobre a hierarquia
```

## Reconhecimentos

A separação em módulos (Dados, Métodos, Avaliação, Relatórios), a interface padronizada de método e o padrão de persistência de resultados brutos para reprocessamento posterior foram inspirados no [TFB](https://github.com/decisionintelligence/TFB):

```
@article{qiu2024tfb,
  title   = {TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods},
  author  = {Xiangfei Qiu and Jilin Hu and Lekui Zhou and Xingjian Wu and Junyang Du and Buang Zhang and Chenjuan Guo and Aoying Zhou and Christian S. Jensen and Zhenli Sheng and Bin Yang},
  journal = {Proc. {VLDB} Endow.},
  volume  = {17},
  number  = {9},
  pages   = {2363--2377},
  year    = {2024}
}
```

HTSBench não replica o agendador paralelo do TFB (`ParallelBackend`) nem seu framework de busca de hiperparâmetros — o escopo deste projeto é a avaliação de métodos sobre séries **hierárquicas**, o que o TFB não cobre.

Os datasets `traffic` e `wiki2`, e a convenção de horizonte=1 usada para ambos, vêm de:

```
@inproceedings{rangapuram2021end,
  title     = {End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series},
  author    = {Rangapuram, Syama Sundar and Werner, Lucien D. and Benidis, Konstantinos and Mercado, Pedro and Gasthaus, Jan and Januschowski, Tim},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  series    = {PMLR},
  volume    = {139},
  pages     = {8832--8843},
  year      = {2021}
}
```

A reconciliação `min_trace`/`min_trace_shrink` segue Wickramasuriya, Athanasopoulos & Hyndman (2019), *Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization*, JASA.

## Contato

Dúvidas, sugestões ou problemas: abra uma [issue](https://github.com/malbuq-dev/hts-bench/issues) neste repositório.
