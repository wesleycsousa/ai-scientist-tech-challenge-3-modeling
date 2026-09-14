# Insights e Aplicação Estratégica

Este documento responde às cinco perguntas de negócio propostas no enunciado, usando o modelo final (Árvore de Decisão, acurácia de validação 0,7315, AUC 0,791) e as técnicas de interpretabilidade aplicadas no notebook `12_interpretabilidade.ipynb`. Todos os números abaixo vêm da execução real desse notebook contra a base completa (42.273 escolas).

Antes de entrar nas perguntas, vale registrar a metodologia: uso três técnicas de interpretabilidade que se complementam. A importância nativa da árvore (Gini) mede a importância por coluna já transformada pelo pipeline, o que significa que uma variável categórica de alta cardinalidade como `sigla_uf` (27 categorias) aparece fatiada em quase 20 colunas de importância individual baixa. A Permutation Importance embaralha a coluna original antes do pré-processamento, então mede o efeito de `sigla_uf` inteira de uma vez. Para confirmar essa leitura, testei também treinar o mesmo classificador com `sigla_uf` codificada como uma única coluna ordinal (em vez de one-hot) e conferir a importância nativa nesse cenário; deixo essa comparação registrada, com a ressalva de que a codificação ordinal impõe uma ordem numérica arbitrária entre as UFs e não é uma codificação melhor, só uma forma diferente de agregar a leitura.

## 1. Quais fatores mais impactam a alfabetização das crianças?

Olhando as três técnicas juntas, o padrão é consistente: o estado onde a escola está (`sigla_uf`) é o fator de maior peso agregado, seguido de perto pela infraestrutura de acesso à tecnologia.

Pela Permutation Importance, `sigla_uf` tem importância de 0,142, mais de dez vezes a segunda colocada (`etapa_ensino_fundamental_anos_finais`, com 0,014, seguida de `acesso_internet_computador`, com 0,013). Pela importância nativa (que fatia `sigla_uf` em uma coluna por UF), a feature isolada mais forte é `acesso_internet_computador` (0,176), seguida por diversas UFs específicas (Paraná 0,074, Ceará 0,070, Bahia 0,068, São Paulo 0,067, Minas Gerais 0,051). Quando testei codificar `sigla_uf` como uma única coluna, ela sozinha chega a 0,310 de importância nativa, a maior de todas as variáveis, o que confirma que as duas métricas não se contradizem: estão só olhando o mesmo fenômeno em granularidades diferentes.

Depois de `sigla_uf` e acesso à internet/computador, os fatores seguintes mais consistentes entre as três técnicas são: se a escola também oferece os anos finais do fundamental (`etapa_ensino_fundamental_anos_finais`), a presença de televisão/equipamento multimídia (`quantidade_equipamento_tv`), a existência de órgão de associação de pais e mestres, e infraestrutura básica de saneamento (esgoto ligado à rede pública, ausência de tratamento de lixo).

Um achado que vale destacar separadamente: escolas que oferecem só os anos iniciais do fundamental (55% da base, 23.247 escolas) batem a meta do IDEB em 60,5% dos casos, contra 47,8% entre as que também oferecem os anos finais (45% da base, 19.002 escolas), uma diferença de quase 13 pontos percentuais. Como o IDEB usado neste projeto é medido especificamente no fim do 5º ano (ciclo "anos iniciais"), essa diferença é compatível com a ideia de que uma escola dedicada só a esse ciclo tende a manter foco mais concentrado nele, enquanto uma escola que atende também do 6º ao 9º ano pode diluir atenção e recursos entre etapas diferentes. É um padrão observado na base, não uma relação de causa comprovada.

Uma leitura individual via SHAP reforça o mesmo ponto: analisei a escola de maior risco previsto pelo modelo (localizada na Bahia) e a escola mais segura (localizada no Paraná). Na escola de maior risco, só o fato de estar na Bahia já reduz a previsão em 0,27 (o maior peso individual isolado observado em qualquer explicação de escola específica), terminando com previsão final próxima de zero. Na escola mais segura, estar no Paraná contribui +0,15 e ter acesso à internet contribui mais +0,09, terminando com previsão final próxima de um. Em ambos os casos, o estado pesa mais que qualquer característica individual de infraestrutura.

## 2. Quais municípios apresentam maior risco educacional?

Agreguei o risco previsto pelo modelo (`risco_previsto`, complemento da probabilidade de bater a meta) por município, considerando só municípios com pelo menos 3 escolas na base para a média não ficar instável por causa de um caso isolado. Os 10 municípios de maior risco médio, todos com risco médio acima de 0,93, são:

| Município (código IBGE) | UF | Risco médio | Escolas na base |
|---|---|---|---|
| 2913804 | BA | 1,000 | 3 |
| 2918308 | BA | 0,959 | 7 |
| 2908903 | BA | 0,953 | 7 |
| 2916609 | BA | 0,953 | 3 |
| 2901700 | BA | 0,946 | 5 |
| 2804300 | SE | 0,943 | 3 |
| 2802007 | SE | 0,943 | 3 |
| 1300060 | AM | 0,942 | 5 |
| 2911501 | BA | 0,939 | 3 |
| 2911857 | BA | 0,939 | 5 |

Oito dos dez municípios de maior risco são da Bahia, com dois municípios de Sergipe e um do Amazonas completando a lista. Isso bate com o padrão de UF mais amplo (próxima pergunta): Bahia, Maranhão, Rio Grande do Norte, Pará e Sergipe são justamente as cinco UFs com maior risco médio previsto na base inteira.

## 3. Existem padrões regionais semelhantes entre diferentes localidades?

Sim, e de forma bem definida. O risco médio previsto por região é:

| Região | Risco médio previsto |
|---|---|
| Norte | 0,683 |
| Nordeste | 0,635 |
| Centro-Oeste | 0,420 |
| Sudeste | 0,301 |
| Sul | 0,243 |

Norte e Nordeste concentram o maior risco, com uma diferença grande frente a Sudeste e Sul (quase o triplo). Esse padrão bate com o que já tinha aparecido na análise exploratória do projeto (diferença de desempenho por região documentada na etapa de EDA), agora confirmado do lado da predição do modelo treinado. Olhando por UF, as cinco mais arriscadas são Bahia (0,801), Maranhão (0,800), Rio Grande do Norte (0,771), Pará (0,756) e Sergipe (0,754), todas do Norte/Nordeste.

Vale registrar que treinei um único modelo nacional. O fato de `sigla_uf` concentrar tanto peso na explicação sugere que cada estado tem uma dinâmica bem diferente de política educacional, investimento e contexto socioeconômico, e que um modelo por estado ou por região (ou um modelo hierárquico, que aprende um padrão nacional e ajusta por estado) provavelmente capturaria melhor os padrões locais. Fica registrado como uma evolução futura natural deste projeto, não como uma falha do modelo atual.

## 4. Como identificar escolas ou municípios que podem não atingir metas educacionais no futuro?

Construí uma classificação de prioridade a partir da probabilidade de risco prevista pelo modelo, aplicada só às escolas que o modelo já classifica como propensas a não bater a meta (`risco_previsto >= 0,50`):

- **Prioridade Alta** (risco >= 0,90): o modelo está bem confiante que a escola não vai bater a meta.
- **Prioridade Média** (0,80 a 0,90).
- **Prioridade Baixa** (0,50 a 0,80): caso limítrofe, ainda do lado "não vai bater", mas com menos confiança do modelo.

Das 42.273 escolas da base, 18.252 (43,2%) entram em alguma faixa de risco: 2.651 em prioridade Alta, 3.396 em Média e 12.205 em Baixa. A distribuição por região confirma o padrão já visto: o Nordeste concentra a maior parte das escolas de prioridade Alta e Média (1.860 e 2.192, respectivamente), seguido do Norte (702 e 920). Sudeste e Sul têm poucas escolas de prioridade Alta (42 e 19, respectivamente), a maior parte das suas escolas em risco cai na faixa Baixa.

Como checagem de sanidade, conferi as 10 escolas de prioridade Alta com maior risco previsto: todas são da Bahia, todas com risco previsto de exatamente 1,0, e todas de fato não bateram a meta na base real (`alvo_real = 0`). Isso indica que o modelo está bem calibrado nesse extremo, e essa lista funciona como um ponto de partida direto para ação de política pública: comece pelas escolas de prioridade Alta.

## 5. Quais variáveis têm mais influência na predição dos modelos?

A resposta é essencialmente a mesma do item 1: `sigla_uf`, como bloco (Permutation Importance 0,142, ou 0,310 quando testada como coluna única), é a variável de maior influência agregada, seguida por `acesso_internet_computador` como a feature individual isolada mais forte entre as demais (0,176 de importância nativa, 0,013 de Permutation Importance). Na sequência aparecem `etapa_ensino_fundamental_anos_finais`, presença de televisão/equipamento multimídia na escola, existência de associação de pais e mestres, e infraestrutura básica de saneamento.

## Limitações a considerar

Vale reforçar duas limitações já discutidas ao longo do projeto. Primeiro, correlação não é causalidade: por exemplo, ter televisão na escola provavelmente funciona como proxy de um pacote maior de investimento e infraestrutura, não como causa direta do desempenho, e o mesmo raciocínio vale para `sigla_uf`, que é mais um proxy de governança e contexto estadual do que uma causa isolada. Segundo, o IDEB usado aqui é o de anos iniciais (1º ao 5º ano, avaliado ao fim do 5º ano), um recorte específico dentro do desempenho educacional mais amplo, com o Censo Escolar de 2024 pareado ao ciclo do IDEB de 2025 como aproximação temporal, e não um match exato ano a ano.
