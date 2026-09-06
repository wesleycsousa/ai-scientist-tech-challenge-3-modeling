# Decisões do Projeto

Registro aqui as decisões de modelagem tomadas ao longo do desenvolvimento, com a justificativa de cada uma. A ideia é que qualquer pessoa lendo este arquivo entenda por que o modelo final ficou do jeito que ficou, não só o que ele faz.

## Dia 1 — Granularidade da base

Decidi partir da tabela `alunos` (Silver), que tem granularidade por aluno avaliado (~3,87 milhões de linhas), em vez das tabelas da Gold, que são todas agregadas por município. O enunciado pede a previsão "se um aluno será alfabetizado", que é um problema de granularidade individual - nenhuma tabela da Gold sozinha atende isso.

Enriqueci a base de `alunos` com um LEFT JOIN em `(id_municipio, ano)` trazendo as colunas de infraestrutura escolar da tabela Gold `indicador_x_infraestrutura_escolar` (água potável, internet, computadores, biblioteca, laboratório de informática, esgoto rede pública). Usei LEFT JOIN porque quero manter todos os alunos no dataset mesmo que o município deles não tenha correspondência na tabela de infraestrutura.

**Importante:** não trouxe as colunas `taxa_alfabetizacao` e `media_portugues` que também existem em `indicador_x_infraestrutura_escolar` - essas métricas são agregações do próprio indicador que estou tentando prever, então trazê-las pro dataset também seria uma forma de data leakage.

## Dia 1 — Data leakage entre `proficiencia` e `alfabetizado`

Testei a hipótese de que `alfabetizado` é derivada diretamente de `proficiencia` pelo corte de 743 pontos do SAEB (esse corte já estava documentado desde a Fase 2 do projeto). Reconstruí a coluna `alfabetizado` aplicando a regra `proficiencia >= 743` e comparei linha a linha com o valor real da coluna.

**Resultado:** a regra bate com o valor real em praticamente 100% das linhas.

**Decisão:** `proficiencia` fica de fora do conjunto de features do modelo. Ela não é uma variável observada de forma independente do alvo - é calculada a partir dele. Se eu deixasse entrar, o modelo não aprenderia os fatores educacionais/territoriais/socioeconômicos que de fato influenciam a alfabetização (que é o objetivo do desafio), ele ia simplesmente reaprender a regra do corte, porque a resposta já estaria embutida na própria feature.

A checagem completa (com o código e os números) está em `notebooks/00_verificacao_leakage.ipynb`.

## Dia 1 — Checkpoint de memória e volume

Medi o tempo de carga e o uso de memória da base completa (alunos + infraestrutura enriquecida): 3.867.999 linhas x 26 colunas, ~1.126 MB de RAM, ~11 segundos para carregar e fazer o join. Como isso é tranquilo pra minha máquina, decidi não otimizar dtypes nem usar amostragem estratificada - sigo com a base completa a partir daqui.
