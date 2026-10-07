# recipe-recsys

Sistema de recomendação de receitas com **restrições alimentares rígidas** (alergias e dietas nunca são violadas), **avaliação offline honesta** e, nas próximas fases, embeddings, uma camada de LLM e deploy.

Dados: [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions), com 231 mil receitas e 1,1 milhão de reviews entre 2000 e 2018.

## Status

| Fase | Conteúdo | Status |
|---|---|---|
| 1 | Pipeline de dados, split temporal, baselines, métricas, filtro de restrições | ✅ |
| 2 | Filtragem colaborativa com embeddings (matrix factorization, two-tower) | ⏳ |
| 3 | Embeddings de conteúdo para cold-start, busca em linguagem natural com LLM, cardápio semanal | ⏳ |
| 4 | API (FastAPI), índice vetorial, Docker, deploy, demo | ⏳ |

## Rodando

Requer [uv](https://docs.astral.sh/uv/). O uv baixa o Python 3.12 automaticamente.

```bash
uv sync
uv run recsys prepare                       # baixa a Food.com e salva em parquet (~1 min)
uv run recsys evaluate --stage val          # ajuste de hiperparâmetros
uv run recsys evaluate --stage test         # números finais
uv run recsys evaluate --stage test --restrict vegetarian gluten
uv run recsys recommend --user 29196 --restrict vegetarian gluten
uv run pytest
```

Para comparar hiperparâmetros: `--models itemknn:neighbors=50,shrink=10.0 itemknn:neighbors=50,shrink=200.0`.

## Protocolo de avaliação

- **Feedback implícito:** cada review conta como "cozinhou a receita". Na Food.com, a nota 0 significa review sem nota, não nota ruim.
- **Split temporal global:** treino até 2011-12, validação até 2014-02 e teste depois disso. Um split aleatório deixaria o modelo aprender com reviews escritas *depois* das que ele precisa prever.
- **k-core (≥ 5 interações por usuário e por receita) aplicado só no treino.** Filtrar com contagens do período de teste vazaria a informação de quais usuários continuam ativos.
- **Validação para ajustar, teste uma única vez.** Os hiperparâmetros foram escolhidos na validação; o teste foi rodado só com as configurações finais.
- **Receitas já cozinhadas no treino são excluídas** das recomendações e do gabarito: o objetivo é recomendar algo novo.
- **Cold-start é reportado, não escondido.** Interações de usuários ou receitas que não existem no treino não podem ser atendidas por filtragem colaborativa, e a fração delas é mostrada em toda avaliação.
- **Métricas:** Recall@K, NDCG@K, HitRate@K (relevância binária) e **cobertura de catálogo**, que é a fração das receitas recomendada a pelo menos um usuário.

## Resultados (teste, 2014-02 → 2018-12)

Treino com 16.873 usuários × 39.476 receitas (densidade de 0,08%). 2.388 usuários avaliados.

| modelo | recall@10 | ndcg@10 | hit_rate@10 | cobertura@10 |
|---|---|---|---|---|
| random | 0.0007 | 0.0007 | 0.0029 | 45.5% |
| popularity | 0.0226 | 0.0140 | 0.0582 | 0.1% |
| recent_popularity (180 dias) | 0.0177 | 0.0121 | 0.0523 | 0.1% |
| itemknn (k=50, shrink=10) | 0.0171 | 0.0129 | 0.0486 | **16.3%** |
| itemknn (k=50, shrink=50) | 0.0218 | 0.0145 | 0.0540 | 4.0% |
| itemknn (k=50, shrink=200) | 0.0220 | **0.0154** | **0.0620** | 1.2% |

Tabelas completas (incluindo @20) em [reports/](reports/).

### O que os números mostram

1. **O baseline de popularidade é difícil de bater.** O ItemKNN mais preciso ganha dele por apenas 10% em NDCG@10. Isso é comum em dados esparsos e é o motivo de nenhum modelo ser reportado sem baselines.
2. **Precisão × diversidade.** A popularidade recomenda as mesmas ~40 receitas para todo mundo (0,1% do catálogo). No ItemKNN, aumentar o `shrink` melhora o NDCG, mas derruba a cobertura de 16% para 1%: o modelo vai *virando* um recomendador de popularidade, porque o shrink favorece itens com muitas coocorrências. A escolha do ponto de operação é de produto, não só de métrica.
3. **A popularidade recente foi pior que a popularidade geral no teste**, embora tenha sido um pouco melhor na validação. O sinal de tendência de 6 meses não se manteve num horizonte de 5 anos de teste.
4. **Cold-start domina.** No teste, **79% das interações vêm de usuários que não existem no treino** e 10% de receitas novas. Só 11% das interações podem ser avaliadas por filtragem colaborativa. É a principal motivação da fase 3: recomendar a partir do *conteúdo* da receita e de preferências declaradas.

## Restrições alimentares

Restrições são um **filtro rígido aplicado depois do ranking**, não um peso no score: uma receita proibida nunca aparece, por mais alto que seja o score.

Restrições disponíveis: `vegetarian`, `vegan`, `lactose`, `egg`, `gluten`, `nuts`.

| restrição (teste) | catálogo permitido | NDCG@10 popularity | NDCG@10 itemknn |
|---|---|---|---|
| vegetarian | 57.5% | 0.0152 | 0.0141 |
| gluten | 45.2% | 0.0205 | 0.0176 |
| lactose + nuts | 33.2% | 0.0248 | 0.0207 |

Decisões de design:

- **Ingredientes, não tags.** As tags da Food.com são preenchidas pelos usuários e não são confiáveis: há receitas marcadas como `gluten-free` que usam molho inglês (worcestershire), que normalmente contém malte de cevada.
- **Conservador por padrão.** Esconder uma receita segura (falso positivo) custa pouco. Mostrar uma insegura (falso negativo) não. Exceções conhecidas, como "coconut milk" e "nutmeg", são liberadas explicitamente em vez de afrouxar as regras.
- **Marcadores valem só para a própria restrição.** "Gluten-free almond flour" é liberado para glúten e continua bloqueado para oleaginosas. "Non-dairy" **não** é um marcador: nos EUA, um produto "non-dairy" pode conter caseinato (proteína do leite).
- **Auditoria contra o próprio dataset.** Listar os ingredientes mais frequentes que cada filtro libera revelou falhas reais na primeira versão: `baguette`, `hamburger buns` e `cream of mushroom soup` passavam pelo filtro de glúten, e `crabmeat` e `catfish` pelo vegetariano (palavras compostas escapam da fronteira de palavra). Cada falha virou um teste de regressão em [tests/test_restrictions.py](tests/test_restrictions.py).

**Limitação importante:** o "violations=0" da avaliação usa o mesmo filtro para auditar, então é uma checagem de consistência, não uma prova de segurança. Palavras-chave não enxergam ingredientes escondidos em produtos industrializados. Um produto real precisaria de dados de ingredientes curados e de um conjunto rotulado para medir a precisão e o recall do filtro. Isso está planejado para a fase 3, comparando com um classificador baseado em LLM.

## Estrutura

```
src/recipe_recsys/
  data.py           download e parsing da Food.com → parquet
  split.py          split temporal e k-core
  dataset.py        matriz esparsa usuário × receita e mapeamento de ids
  restrictions.py   restrições alimentares (filtro rígido)
  metrics.py        recall, ndcg, hit rate, cobertura
  evaluate.py       protocolo de avaliação, máscaras e top-k
  models/           random, popularity, recent_popularity, itemknn
  cli.py            recsys prepare | evaluate | recommend
tests/              métricas, split sem vazamento, restrições e regressões
reports/            resultados das avaliações (.md versionado, .json ignorado)
```

Todos os modelos implementam a mesma interface (`fit` e `score`). As máscaras (itens já vistos e restrições) e a seleção do top-k ficam fora do modelo, para que todos recebam exatamente o mesmo tratamento na avaliação.

Uma nota de engenharia: a matriz de coocorrência item × item do ItemKNN tem cerca de 10⁸ entradas. Ela é construída em blocos de linhas, podados para os top-k vizinhos antes do bloco seguinte, o que mantém o pico de memória em cerca de 1,5 GB.
