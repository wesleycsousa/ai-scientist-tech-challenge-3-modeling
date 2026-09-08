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

A checagem completa (com o código e os números) está em `notebooks/eda/00_verificacao_leakage.ipynb`.

## Dia 1 — Checkpoint de memória e volume

Medi o tempo de carga e o uso de memória da base completa (alunos + infraestrutura enriquecida): 3.867.999 linhas x 26 colunas, ~1.126 MB de RAM, ~11 segundos para carregar e fazer o join. Como isso é tranquilo pra minha máquina, decidi não otimizar dtypes nem usar amostragem estratificada - sigo com a base completa a partir daqui.

## Dia 2 — Compilado dos achados da Análise Exploratória (EDA)

Este bloco reúne tudo que apurei e discuti até agora na EDA (`notebooks/eda/01_eda.ipynb`), pra servir de referência única antes de fechar as decisões finais de modelagem. Separei o que já é uma decisão fechada do que ainda está em aberto (o detalhamento das pendências está em `doc/pendencias.md`).

### Achado 1 — Desbalanceamento de classes é leve

`alfabetizado`: 51,31% classe 1, 48,69% classe 0. Não há necessidade de balanceamento artificial (SMOTE, `class_weight`, etc.). Um classificador bobo que sempre prevê a classe majoritária atinge 51,31% de acurácia — esse é o piso que qualquer modelo real precisa superar por uma margem clara.

**Decisão:** métrica principal de avaliação no Dia 4 = AUC-ROC + F1 da classe "não alfabetizado" (mais custoso, num contexto de política pública, deixar de identificar quem está em risco), com acurácia reportada só como complemento.

### Achado 2 — Infraestrutura escolar tem importância baixa no diagnóstico, mas isso tem uma explicação estrutural

No diagnóstico de feature importance (Seção 8 da EDA), as colunas `pct_escolas_*` tiveram peso individual pequeno. Isso provavelmente não significa que infraestrutura "não importa" — é herança de uma limitação da base: esse dado é uma média por **município** (todas as escolas do município misturadas, vindo do Censo Escolar agregado já na Fase 2), não por escola individual (`id_escola`). Como todo aluno do mesmo município recebe exatamente o mesmo valor, o sinal real "borra" — se existisse o dado por escola, provavelmente teria mais poder explicativo. Documentado como limitação conhecida herdada da Fase 2, fora do escopo resolver agora (exigiria voltar ao Censo Escolar bruto por escola).

Multicolinearidade entre as colunas de infraestrutura é baixa (correlação de Spearman máxima de 0,43 entre `pct_escolas_esgoto_rede_publica` e `pct_escolas_laboratorio_informatica`) — não é motivo para remover nenhuma delas.

### Achado 3 — `rede` de ensino: correlação real, mas correlação ≠ causalidade

Taxa de alfabetização por rede: Privada 64,00%, Estadual 53,15%, Municipal 51,07%. Apesar da diferença real, `rede` teve importância baixíssima no modelo diagnóstico (0,0054) — não é contradição: em modelos de árvore, importância mede o ganho marginal *além* do que outras variáveis correlacionadas já explicam, não a associação bivariada isolada.

Na Seção 13 da EDA, testei a hipótese de confounding: comparei a infraestrutura média do município entre alunos de cada rede. Resultado:

| Rede | Taxa alfabetização | Água potável | Internet | Equip. computador | Biblioteca | Lab. informática | Esgoto rede pública |
|---|---|---|---|---|---|---|---|
| Privada | 64,00% | 100,00% | 100,00% | 98,40% | 53,20% | 40,30% | 50,00% |
| Estadual | 53,15% | 98,53% | 95,71% | 92,39% | 44,57% | 37,59% | 74,77% |
| Municipal | 51,07% | 97,55% | 95,24% | 91,08% | 43,18% | 32,66% | 61,48% |

**Leitura:** em 5 das 6 colunas de infraestrutura (água potável, internet, equipamento de computador, biblioteca, laboratório de informática), a ordem Privada > Estadual > Municipal é **exatamente a mesma** ordem da taxa de alfabetização — evidência consistente de que a associação de `rede` com `alfabetizado` está, ao menos em parte, confundida com a condição geral de infraestrutura (e provavelmente socioeconômica) do município onde o aluno estuda, não sendo um efeito isolado da rede em si.

**Mas atenção a uma exceção que evita uma conclusão simplista demais:** `esgoto_rede_publica` **inverte** completamente esse padrão (Estadual 74,77% > Municipal 61,48% > Privada 50,00% — a rede privada é a *pior* nessa dimensão específica). Isso mostra que "infraestrutura" não é um eixo único de "município rico vs. pobre" — conectividade à rede pública de esgoto pode depender mais de zoneamento urbano/histórico de saneamento do município do que da condição socioeconômica geral (hipótese, não confirmada com os dados que tenho). Ou seja, o confounding é real e relevante, mas não é uma explicação completa e mecânica — não dá pra reduzir toda a diferença de `rede` a "infraestrutura municipal".

**Decisão:** `rede` continua como feature (a associação é real e relevante pra descrever o fenômeno), mas o README e a interpretação final devem deixar explícito que essa relação é majoritariamente correlacional/confundida com a condição do município (5 de 6 indicadores de infraestrutura seguem a mesma ordem), não um efeito causal isolado de "estudar em escola privada". O enriquecimento com o Atlas do Desenvolvimento Humano (IDH-M/renda por município, Camada 2, ainda não implementado) é o que vai permitir separar melhor esses efeitos — com a ressalva de que, mesmo assim, qualquer variável de município é uma correlação ecológica, não uma medida direta da condição socioeconômica da família do aluno.

### Achado 4 — Padrão regional existe, mas não tinha sido testado dentro do modelo

Taxa de alfabetização por região: Centro-Oeste 54,71%, Sudeste 53,57%, Sul 52,79%, Nordeste 50,65%, Norte 41,77% — diferença relevante (quase 13 p.p. entre extremos). Esse padrão não aparecia no diagnóstico de feature importance porque `nome_regiao` simplesmente não tinha entrado como feature testada (só tinha sido olhado de forma bivariada). Corrigido: agora a Seção 8 da EDA inclui região via one-hot encoding entre as features de município.

### Achado 5 — Ano (2023 vs. 2024) não muda o padrão de forma relevante

Taxa de alfabetização: 50,21% em 2023, 52,21% em 2024 — diferença pequena (~2 p.p.), não justifica modelar os anos separadamente. Atenção: `taxa_alfabetizacao_ano_anterior` e `variacao_pp` só existem pra linhas de 2024 (2023 não tem "ano anterior" dentro da base — só temos 2 anos de dado granular).

### Achado 6 (crítico) — Leakage estrutural confirmado em `presenca` e `preenchimento_caderno`

Na Seção 9 da EDA, testei se `alfabetizado` é definida de forma mecânica a partir dessas duas colunas de metadado da aplicação da prova. Resultado da tabela cruzada:

| Categoria | % alfabetizado=0 | % alfabetizado=1 |
|---|---|---|
| Ausente | 100,00% | 0,00% |
| Presente | 40,86% | 59,14% |
| Prova não preenchida | 100,00% | 0,00% |
| Prova preenchida | 40,84% | 59,16% |

**Confirmado:** 100% dos alunos ausentes (ou com prova não preenchida) são classificados como `alfabetizado=0`, sem exceção. Isso não é uma coincidência estatística nem um efeito causal de "estar presente" — é a própria regra de geração do indicador: sem presença e sem prova preenchida não existe `proficiencia` medida, então o aluno é automaticamente contado como não alfabetizado (consistente com a metodologia do Compromisso Nacional Criança Alfabetizada, que só confirma a alfabetização de quem foi avaliado).

Isso é estruturalmente parecido com o leakage de `proficiencia` (Dia 1), mas com uma diferença importante: lá, a regra determinava 100% das linhas; aqui, a regra só é determinística dentro do subgrupo "ausente"/"não preenchido" — dentro do subgrupo "presente"/"preenchido", existe uma distribuição real (59,14%/40,86%) que reflete os fatores educacionais/territoriais que quero modelar. Ou seja, não é leakage "geral" da base, é leakage **condicional a um subgrupo específico**.

Isso também explica por que essas duas colunas dominaram a importância do modelo diagnóstico na Seção 8 (mais de 70% somadas): o modelo estava, em boa parte, só "decorando" essa regra de ausência, não aprendendo sobre os fatores reais de alfabetização.

**Deep dive (Seção 9.1):** fiz duas checagens antes de decidir o tratamento. Primeiro, se as duas colunas são redundantes entre si: sim, quase totalmente — dos 512.153 alunos ausentes, 100% têm prova não preenchida (óbvio, quem não compareceu não preenche nada); a única divergência real são 1.185 alunos presentes que mesmo assim não preencheram (0,03% da base). Ou seja, `preenchimento_caderno` não acrescenta quase nada que `presenca` já não diga sozinha. No total, **513.338 linhas (13,27% da base)** são afetadas pela regra mecânica — uma fatia relevante, mas ainda deixa ~86,7% (~3,35 milhões de alunos) disponíveis se eu decidir restringir o escopo.

Segundo, se a não-participação tem um padrão territorial: tem, e curiosamente **não repete o mapa geográfico da alfabetização**. Por região, a taxa de não-participação é Sul 18,12% > Norte 17,98% > Sudeste 12,55% > Centro-Oeste 11,75% > Nordeste 9,16% — note que o Sul (boa alfabetização) tem a maior não-participação, e o Nordeste (alfabetização mediana-baixa) tem a menor. Por rede, o padrão é mais intuitivo: Estadual 14,42% > Municipal 13,13% > Privada 4,00% (rede privada com participação bem mais garantida). Registro isso como um achado à parte — não resolve a pendência de tratamento dessas colunas no modelo de alfabetização, mas é candidato a virar uma pergunta de negócio do Dia 6 (mapa de não-participação como indicador complementar de risco educacional).

**Decisão sobre o tratamento:** ainda não fechada — registrada em `doc/pendencias.md` (envolve escolher entre excluir as colunas, restringir o escopo do modelo a alunos presentes/com prova preenchida, ou manter como parte legítima da definição oficial do indicador, com a devida documentação). Também percebi que o pequeno grupo de alunos sem dado de infraestrutura (3.209 alunos, 0,1% da base) tem taxa de alfabetização de exatamente 0% (Seção 12) — cheguei a especular que pudesse ser o mesmo fenômeno do grupo de leakage, mas a escala não bate (3.209 é ~160x menor que os 513.338 afetados pela regra de ausência) — na melhor hipótese seria um subconjunto pequeno, não o mesmo grupo. Ainda precisa de checagem direta (ver pendências).

### Achado 7 — `taxa_alfabetizacao_ano_anterior` e `variacao_pp`: não é o mesmo tipo de problema que `presenca`/`preenchimento_caderno`

Essas duas colunas vêm da tabela Gold `evolucao_temporal` (granularidade `(nivel_geografico, local_id, ano)`, filtrada em `rede="5"` — Pública Estadual+Municipal, ver handoff seção 8.9), trazidas via `ler_evolucao_temporal_municipio()`. São dados de **município**, não de aluno — todo aluno do mesmo município no mesmo ano recebe o mesmo valor. Existem só pra 2024 (2023 fica nulo, porque não há um "ano anterior" dentro da janela de dados que temos).

Diferente de `presenca`/`preenchimento_caderno`, isso **não é leakage** no sentido de vazar informação do próprio resultado que estou tentando prever: é uma informação histórica genuína, que já existiria no momento da previsão (o resultado de 2023 é conhecido antes de eu precisar prever 2024). É o mesmo princípio de um modelo de série temporal usar o valor do período anterior como feature.

Ainda assim, tem ressalvas a documentar: (1) é uma agregação do próprio fenômeno que estou prevendo, só que defasada um ano e filtrada pra rede pública — quando aplicada a um aluno de rede privada, por exemplo, está descrevendo a taxa da rede pública do município dele, não da rede dele; (2) por carregar tanto sinal, ela tende a dominar a importância do modelo e a "mascarar" o efeito de outras variáveis territoriais/socioeconômicas, empurrando a interpretação do modelo pra mais perto de "o município já era bom/ruim historicamente" do que pra fatores estruturais explicáveis; (3) essencialmente já é uma versão simples da ideia de "score de município" discutida no `notebooks/eda/02_exploracao_score_municipio.ipynb`. Decisão de manter, ressalvar ou não usar: registrada como pendência.

### Achado 8 — GroupKFold é necessário, não é só cautela teórica

Medido na Seção 10 da EDA: um split aleatório comum (`train_test_split`) deixou **99,5%** dos municípios do conjunto de teste também presentes no treino. Um `GroupShuffleSplit` por `id_municipio` reduziu isso a **0,0%** (por construção). Isso confirma, com número real, que sem uma estratégia de split por grupo eu estaria validando o modelo em municípios que ele já "viu" durante o treino — o que infla artificialmente qualquer métrica de validação.

**Decisão:** usar `GroupKFold`/`GroupShuffleSplit` por `id_municipio` na validação do Dia 3, não um split ou `StratifiedKFold` comum.

### Achado 9 — Ausência de dado de infraestrutura é rara e parece estar ligada ao grupo de ausentes

Apenas 0,1% dos alunos (3.209) não têm dado de infraestrutura do município — e esse grupo tem taxa de alfabetização de exatamente 0,00% (Seção 12), o que bate com o padrão do Achado 6 e sugere que pode ser o mesmo subconjunto de alunos ausentes/prova não preenchida (a confirmar). Ainda não fechei a estratégia de imputação — registrado em pendências.

## Dia 2 — Decisão de arquitetura: reformulação da granularidade para ESCOLA

Esta é a decisão mais importante do projeto até agora, então documento com bastante detalhe.

**Contexto que levou à decisão:** a EDA (Achados 1-9 acima) mostrou de forma consistente que a tabela `alunos` (Silver), que tem a granularidade "correta" segundo a leitura literal do enunciado ("prever se um aluno será alfabetizado"), é pobre em features genuinamente individuais: só `serie`, `rede`, `presenca` e `preenchimento_caderno` existem nesse nível, sem nenhuma variável demográfica (idade, sexo, raça, renda da família). Depois de confirmar que `presenca`/`preenchimento_caderno` são leakage estrutural (Achado 6) e que `serie` tem importância zero, praticamente não sobra sinal individual — o que sobra de poder preditivo é quase todo herdado do município (infraestrutura, metas, região), repetido de forma idêntica para todos os alunos do mesmo município/ano.

Para confirmar se esse era mesmo um limite dos *dados disponíveis* (e não só da minha implementação), consultei o catálogo completo de dados da Fase 2 (`catalogodadosfase3.md`, documento próprio que fiz o levantamento das tabelas Bronze/Silver/Gold e suas colunas). O catálogo confirmou a suspeita e trouxe uma informação nova e decisiva: a camada **Bronze** tem uma tabela `br_inep_censo_escolar.escola` **já ingerida** (não precisa baixar nada novo), com granularidade de **escola individual**, **455 colunas**, incluindo:
- Infraestrutura detalhada por tipo (água/energia/esgoto/lixo, não só um % agregado como na Silver que eu já usava)
- Espaços pedagógicos, acessibilidade, equipamentos/tecnologia, corpo técnico-administrativo
- **Matrículas por sexo, raça/cor, faixa etária, zona de residência e turno** — exatamente a demografia que falta em `alunos` e que eu não tinha
- Uma coluna `programa_brasil_alfabetizado` (flag de participação num programa ligado diretamente ao tema)

Outro ponto técnico importante: a tabela Silver `escola_por_municipio` que eu já vinha usando (agregada por município) foi construída a partir dessa mesma tabela Bronze `escola` — ou seja, meu programa de infraestrutura por município era uma agregação de um dado que originalmente é por escola. Reprocessar por `id_escola` em vez de `id_municipio` é esforço baixo (reaproveita o que já está ingerido), não uma nova coleta de dados.

**As três opções avaliadas** (documentadas no catálogo, seção 1):
- **A — Aluno puro (o que eu tinha até aqui):** ~3,9M linhas, só 4-5 features fracas de aluno + contexto diluído por município. Mais literal ao enunciado, mas é a opção mais pobre analiticamente — exatamente o que a EDA vinha mostrando.
- **B — Aluno + escola (híbrida):** mantém 1 predição por aluno, mas cada aluno herda o contexto real da própria escola (455 colunas) via JOIN em `id_escola`, em vez da média diluída do município inteiro.
- **C — Escola pura:** muda a unidade de predição para a escola (ex.: "a escola terá um bom percentual de alunos alfabetizados"), com o volume caindo de milhões de alunos para ~215 mil escolas (ainda um volume grande).

**Decisão tomada:** Opção **C — remodelar o problema para o nível de escola**.

**Justificativa:** a escola, além de ter de longe a base de features mais rica disponível nos dados (sem precisar de nenhuma ingestão nova), é a unidade sobre a qual políticas públicas de fato conseguem atuar diretamente — de forma mais operacionalizável que "o aluno individual" (o gestor público não consegue intervir em um aluno específico com a mesma facilidade que consegue investir/intervir numa escola) e mais granular que "o município" (que mistura escolas muito diferentes entre si, como já mostrou a Seção 13 da EDA). Esse tipo de decisão — traduzir o pedido original do stakeholder ("prever o aluno") para uma formulação de dados mais adequada e acionável, com base no que os dados realmente sustentam — se chama **reformulação (reframing) do problema**, uma prática reconhecida e valorizada em ciência de dados, não um desvio do que foi pedido. Pretendo documentar esse raciocínio explicitamente no README final (Dia 7), incluindo a Opção A como "abordagem inicial testada e superada" — isso demonstra o processo analítico completo, não só o resultado final.

**Limitação assumida por causa dessa escolha:** a tabela Bronze `escola` só foi ingerida para `ano=2024` (não existe para 2023). Decisão: aplicar os dados de 2024 também aos alunos avaliados em 2023 — a mesma premissa que a Fase 2 já usa para a infraestrutura agregada por município ("infraestrutura não muda muito ano a ano"), agora estendida ao nível de escola. Isso preserva as duas safras de alunos (2023 e 2024) no cálculo do alvo por escola, ao custo de assumir que a estrutura física/de equipe da escola não mudou significativamente entre os dois anos — premissa razoável dado o intervalo curto (1 ano), mas que precisa ficar documentada como limitação no README final.

**Impacto no que já foi feito:** a EDA da Opção A (`notebooks/eda/01_eda.ipynb`, Seções 1-14) não é descartada — ela é exatamente a evidência que fundamenta essa decisão, e continua sendo commitada como parte do Dia 2. As pendências que dependiam da granularidade aluno/município (uso de `taxa_alfabetizacao_ano_anterior`, arquitetura de "score de município") ficam reavaliadas à luz da nova granularidade em `doc/pendencias.md`.

**Próximos passos que essa decisão abre** (não decididos ainda, novas pendências):
- Definir a variável-alvo por escola (ex.: % de alunos alfabetizados por `(id_escola, ano)`, e o corte para "escola boa"/"escola ruim" — mediana? quartil? meta nacional de referência?)
- Checar a interseção entre escolas avaliadas (`alunos`) e escolas no Censo Escolar (`escola`) via `id_escola` — nem toda escola do Censo tem alunos avaliados nos anos iniciais, e vice-versa
- Selecionar e tratar as features relevantes dentre as 455 colunas (muitas são contagens brutas que precisam virar proporções/razões — ex. razão aluno/professor)
- Reaplicar a mesma disciplina de verificação de leakage já usada na Opção A (ex.: cuidado com colunas que sejam agregações do próprio alvo)

### Exploração concluída: modelo de "score de município" em duas etapas — não seguida

Cheguei a registrar em `notebooks/eda/02_exploracao_score_municipio.ipynb` a ideia de um modelo auxiliar que pontua o "risco" do município e alimenta isso como feature do modelo principal por aluno. A checagem preliminar de viabilidade (correlação de Spearman de 0,667 entre a taxa de alfabetização de um mesmo município em 2023 e 2024) confirmou que existe sinal real nessa ideia, não seria só ruído. Ainda assim, essa arquitetura de duas etapas deixou de ser necessária: o projeto pivotou para o grão de escola, com o IDEB como alvo externo (ver seção "Dia 3" abaixo) — no grão de escola, características de território e histórico entram diretamente como features da própria escola, sem precisar de um modelo auxiliar em duas etapas. Fica registrado como uma alternativa investigada e descartada, com a justificativa documentada, não como uma pendência em aberto.

## Dia 3 — Da falta de alvo por escola ao IDEB como variável-alvo definitiva

A decisão de mudar a granularidade para escola (Dia 2) resolveu o problema de features pobres, mas abriu um problema novo: a base de escola não tem, de cara, uma variável-alvo. Esta seção documenta a investigação que levou até uma variável-alvo real, validada e sem leakage — incluindo os obstáculos que apareceram no meio do caminho, porque acho que eles fazem parte da decisão tanto quanto o resultado final.

### Achado 10 — A tabela Bronze de escola só trouxe 11 colunas, não as 455 esperadas

Ao reprocessar por `id_escola`, a leitura real da tabela Bronze `br_inep_censo_escolar.escola` trouxe só 11 colunas — não as 455 que o catálogo de dados descrevia. Antes de assumir que a decisão de pivotar pra escola tinha perdido a base, investiguei o S3 diretamente (arquivos, tamanhos, pastas vizinhas) e confirmei que o catálogo descrevia o schema teórico disponível na fonte, mas só uma ingestão parcial tinha sido feita pro S3. Resolvido com uma nova ingestão completa da tabela (`escola_completo`, path `bronze/br_inep_censo_escolar/escola_completo/`), confirmada com as 455 colunas reais.

### Achado 11 — `id_escola` não é compatível entre `alunos` e `escola`: dois sistemas de código diferentes

O JOIN entre `alunos` (grão de aluno) e `escola`/`escola_completo` (grão de escola) por `id_escola` deu 0% de interseção nos dois sentidos. Um zero absoluto desse tipo não é "interseção baixa", é sinal de erro de chave. Investigando o formato bruto dos dois lados: `escola`/`escola_completo` usa o código INEP oficial (prefixo de 2 dígitos = código de UF do IBGE, 11 a 53); já 100% das ~3,87 milhões de linhas de `alunos` têm `id_escola` começando em `60`, que não corresponde a nenhuma UF válida. Conclusão: são dois sistemas de codificação diferentes, não um problema de formatação. Esse crosswalk específico nunca foi consertado diretamente — o que resolveu o problema, na prática, foi abandonar essa junção e buscar a variável-alvo em outra fonte (Achado 14).

### Achado 12 — `escola_completo` não tem nenhuma variável-alvo nativa

Com as 455 colunas confirmadas, busquei por palavra-chave (aprovação, reprovação, rendimento, IDEB, proficiência, desempenho, nota, taxa, SAEB, evasão, etc.) qualquer indicador de resultado educacional dentro da própria base de escola. Não encontrei nenhum — a única coluna com relação direta ao tema, `programa_brasil_alfabetizado`, é uma flag de participação em programa, não uma medida de desempenho. A `escola_completo` descreve muito bem características de entrada (infraestrutura, corpo docente/técnico, matrícula, localização, gestão), mas não tem nenhuma variável de saída/resultado.

### Achado 13 — Viabilidade de recuar pro grão de município foi considerada e descartada

Diante da falta de alvo nativo por escola, cheguei a levantar a alternativa de recuar pro grão de **município** (onde `id_municipio` já funciona como chave de JOIN confiável, diferente de `id_escola`) e agregar a `escola_completo` inteira por município, ganhando bem mais features do que as poucas `pct_escolas_*` que já usava. Não cheguei a terminar de rodar essa checagem de viabilidade — antes disso, a busca por uma fonte externa de desempenho por escola (Achado 14) resolveu o problema diretamente no grão de escola, tornando essa alternativa desnecessária. Fica descartada com justificativa registrada, não como pendência aberta: o grão de escola é mais acionável que o de município (ver a seção de reflexão, mais abaixo).

### Achado 14 — IDEB como variável-alvo externa, com crosswalk validado

Pesquisando fontes externas de indicador de desempenho por escola, encontrei o dataset `br_inep_ideb`, que traz o **IDEB** (Índice de Desenvolvimento da Educação Básica) calculado pelo INEP a nível de escola. Antes de usar, confirmei com dado real — e não assumido — que o `id_escola` desse dataset bate com o da `escola_completo`: interseção de **95,89%** das escolas do `ideb_escola`, um crosswalk real e utilizável, bem diferente do que aconteceu com `alunos` (Achado 11).

**Decisão da variável-alvo:** `alvo = 1 se ideb >= 6.0, senão 0`. O corte de 6,0 não é arbitrário — é a meta nacional de referência do próprio MEC/INEP (aproximadamente a nota dos sistemas educacionais mais bem colocados no PISA).

### Achado 15 — Checagem de leakage do próprio IDEB

Antes de aceitar o IDEB como alvo, precisava garantir que ele não incorporasse informação de infraestrutura (o que criaria leakage com as próprias features que pretendo usar). A documentação oficial do INEP confirma que `ideb = indicador_rendimento × nota_saeb_media_padronizada` — só fluxo escolar e desempenho na prova, sem infraestrutura, matrícula ou corpo docente. Recalculei a fórmula em 787.703 linhas reais e comparei com o valor já existente na tabela: diferença média de 0,025 (numa escala de 0 a 10), consistente com arredondamento em cascata do INEP nas etapas intermediárias do cálculo, não com um terceiro fator escondido.

**Conclusão:** infraestrutura (`escola_completo`) e desempenho (`ideb`) são fontes independentes — não há leakage entre elas. O que fica de fora das features, por compor o próprio alvo, é `taxa_aprovacao`, `indicador_rendimento`, as notas do SAEB, `ideb` e `projecao`.

### Achado 16 — Exclusão da rede privada

Decisão: filtrar o modelo só para rede pública (municipal + estadual + federal). Dois motivos: (1) dado — a rede privada tem preenchimento muito pior de `taxa_aprovacao`/`ideb` (19,90%/13,67% contra 70,32%/61,64% da pública) e representa menos de 0,5% da base; no recorte de anos iniciais ela nem aparece; (2) conceitual — o objetivo do projeto é apoiar política pública de alfabetização, e a rede privada tem outros meios (recursos próprios, mercado) de identificar e corrigir problemas de aprendizagem. O valor de um modelo preditivo é maior justamente onde a gestão tem menos recursos de monitoramento próprio.

### Achado 17 (decisão final) — Alinhamento de ano: Censo 2024 pareado com IDEB 2025

A `escola_completo` (Censo Escolar) só existe para `ano = 2024` — confirmado com `value_counts()` real sobre as 215.545 linhas, 100% do mesmo ano, sem mistura. O IDEB só é calculado em anos ímpares (2005 a 2025), então não existe um "match exato" de ano entre as duas fontes.

**Decisão final: uso o ciclo do IDEB de 2025 pareado com a infraestrutura do Censo de 2024.** Essa escolha se apoia em três pontos:

1. **Preenchimento:** 2025 tem o maior percentual de preenchimento de `ideb` entre os ciclos recentes (71,08%).
2. **Lógica causal:** 2025 é posterior ao Censo de 2024, o que é mais defensável do que usar um ciclo anterior (infraestrutura registrada antes → resultado de um ciclo de avaliação seguinte) — coerente com a pergunta de negócio de identificar escolas que podem não atingir metas futuras. A alternativa mais próxima "antes" do Censo seria 2023 (67,44% de preenchimento), que ficaria logicamente mais estranha de justificar (features de 2024 "depois" do resultado que quero prever).
3. **Estabilidade da infraestrutura escolar ano a ano:** características de infraestrutura (água potável, internet, biblioteca, laboratório, esgoto) não mudam de um ano pro outro com a mesma velocidade que um indicador de desempenho muda. Construir uma biblioteca ou levar internet pra uma escola é um investimento que, uma vez feito, tende a se manter por vários anos — diferente de um resultado de avaliação, que pode oscilar mais de um ciclo pro outro. Por isso, a foto de infraestrutura de 2024 ainda descreve razoavelmente bem a escola em 2025. Essa é a mesma lógica que a Fase 2 já usava para justificar infraestrutura agregada por município como estável ano a ano (Dia 2) — agora estendida, com o mesmo raciocínio, ao alinhamento entre o Censo Escolar e o ciclo do IDEB.

### Achado 18 — Sanity check manual confirma que os dados são reais e localizáveis

Selecionei uma amostra de escolas municipais com IDEB preenchido e conferi manualmente uma delas (`id_escola = 26034999`) na consulta pública do INEP/QEdu: existe de verdade, é a "Escola Municipal Santo Antônio", em Pernambuco, com rede e UF batendo com a minha base. Confirma que os `id_escola` são códigos INEP reais, não códigos inventados ou de outra fonte.

### Achado 19 (ponto de atenção ético) — proxy racial na correlação com o alvo

Na EDA de correlação das 455 colunas da `escola_completo` contra `ideb` (`notebooks/eda/07_eda_escola_silver.ipynb`), a variável de maior correlação absoluta com o alvo em toda a base foi `quantidade_matricula_branca` (contagem de matrículas de alunos autodeclarados brancos), com Spearman de 0,403 — mais forte que qualquer variável de infraestrutura. Isso não é uma relação causal: é reflexo de um problema bem documentado no Brasil, onde composição racial da escola correlaciona com contexto socioeconômico e histórico de desigualdade regional/urbana, que por sua vez afeta desempenho. Uso essa variável só para descrever o padrão nesta EDA.

**Decisão registrada como pendência:** usá-la ou não como feature direta do modelo é uma questão sensível — incluir sem tratamento pode fazer o modelo aprender um proxy racial em vez de um fator realmente acionável por política pública (como infraestrutura ou corpo docente). Fica para ser resolvida explicitamente na hora de fechar a seleção final de features, não decidida dentro da EDA.

## Reflexão — o processo de reframing deste projeto, do início ao fim

Registro aqui, num bloco só e de forma mais reflexiva do que o resto deste documento, o processo de reformulação (reframing) que atravessou este projeto do Dia 1 ao Dia 3. Não é uma decisão pontual como as anteriores — é o fio que costura praticamente todas elas, e por isso acho que merece uma seção própria, separada da lista de achados numerados. Fica registrado aqui porque é, na minha visão, o aprendizado mais importante de toda essa etapa: não é sobre qual algoritmo escolher, é sobre garantir que a pergunta que o modelo vai responder é a pergunta certa, com o dado certo por trás dela.

### O caminho, resumido

O enunciado do desafio pede um modelo que preveja "se um aluno será alfabetizado". Lido ao pé da letra, isso aponta pra um modelo com granularidade de aluno — foi exatamente onde comecei (`notebooks/eda/01_eda.ipynb`). A base de aluno tem quase 3,9 milhões de linhas, e à primeira vista parece o dado mais "correto" para o problema, porque é literalmente a unidade que o enunciado menciona.

O que a EDA mostrou, com número real e não com achismo, foi que essa base é pobre em informação genuinamente do aluno: só `serie`, `rede`, `presenca` e `preenchimento_caderno` existem nesse grão — e as duas últimas, como registrei no Achado 6, são leakage estrutural (a regra de geração do próprio indicador, não um fator educacional real). Sem elas, praticamente todo o poder preditivo que sobrava vinha de características herdadas do **município** — repetidas de forma idêntica para todos os alunos do mesmo município e ano, porque a camada Gold que eu tinha herdado da Fase 2 já vinha agregada nesse nível.

Isso levou a uma primeira tentativa de contorno, sem abandonar o aluno como unidade de predição: montar um "score de município" auxiliar, numa arquitetura em duas etapas (`notebooks/eda/02_exploracao_score_municipio.ipynb`). A checagem preliminar confirmou que existe sinal real nisso (correlação de Spearman de 0,667 entre o resultado de um mesmo município em anos consecutivos) — mas era, no fundo, um jeito de contornar o sintoma sem resolver a causa raiz: a informação que eu tinha estava agregada demais para o que eu precisava.

### Por que a camada agregada (Gold) não é suficiente para modelar bem

Esse é o ponto central de todo o reframing, e vale destacar de forma direta: **uma camada de dados agregada por município, por mais bem construída que seja, não serve bem como base de um modelo que precisa diferenciar unidades menores do que o município** (sejam alunos, sejam escolas). O problema não é a qualidade do dado agregado em si — é que a agregação, por definição, elimina a variação que existe dentro do grupo. Todo aluno do mesmo município recebe exatamente o mesmo valor de "% de escolas com biblioteca", por exemplo, mesmo que a escola dele especificamente tenha ou não biblioteca. Isso "borra" o sinal real: uma escola com ótima infraestrutura e outra com infraestrutura péssima, no mesmo município, ficam estatisticamente indistinguíveis nessa camada — quando na prática são situações bem diferentes, com implicações de política pública bem diferentes.

Foi só indo atrás de uma informação mais granular do que a que já estava pronta (a tabela Bronze de escola individual, com 455 colunas, ainda não trabalhada na Fase 2) que consegui, de fato, um caminho pra sair desse limite. Esse trabalho de investigação não foi trivial nem rápido: veio acompanhado de um schema que não batia com o documentado (a tabela real trouxe só 11 colunas na primeira tentativa, não as 455 esperadas — resolvido com uma nova ingestão), de um crosswalk de identificador que se revelou quebrado (`id_escola` de `alunos` usa um sistema de código diferente do `id_escola` oficial do INEP, uma incompatibilidade que só apareceu porque fui conferir o dado bruto dos dois lados, em vez de assumir que "deveria bater"), e da constatação de que nem essa base granular tinha uma variável-alvo pronta (a `escola_completo` descreve muito bem entrada, mas não tem nenhuma coluna de resultado educacional). Cada um desses obstáculos exigiu voltar pro dado bruto e confirmar com número real antes de seguir — a mesma disciplina que já vinha usando desde o Dia 1 pra não decidir por achismo.

### Entender o problema antes de modelar

Um bom modelo não nasce de sair rodando `fit()` na primeira base disponível — nasce de entender de verdade o problema que está sendo resolvido, o que os dados realmente significam, e onde eles podem estar mentindo. Boa parte do tempo desta etapa não foi gasta ajustando hiperparâmetro nenhum: foi gasta verificando se um identificador realmente identifica a mesma coisa nos dois lados de um JOIN, se uma variável de resultado é calculada de forma independente das features que pretendo usar, se uma métrica de preenchimento é boa o suficiente pra sustentar uma conclusão, se um zero absoluto de interseção é um achado real ou um sinal de erro de chave. Isso é trabalho de ciência de dados tanto quanto treinar um modelo — na verdade, sem esse trabalho, qualquer modelo treinado em cima estaria respondendo a uma pergunta equivocada, ou pior, aprendendo um atalho que não existe na vida real, que é exatamente o que os achados de leakage, descritos a seguir, mostram.

### Data leakage: um cuidado recorrente, não um evento isolado

O cuidado com vazamento de dado apareceu em praticamente toda etapa deste projeto, não uma vez só:

- No Dia 1, confirmei que `proficiencia` reproduz `alfabetizado` por uma regra de corte determinística (743 pontos do SAEB) — usá-la como feature seria o modelo "decorar" a resposta em vez de aprender fatores reais.
- Também no Dia 1 (Achado 6), descobri que `presenca` e `preenchimento_caderno` têm uma relação mecânica com o alvo dentro de um subgrupo específico (quem está ausente é automaticamente `alfabetizado=0`) — um leakage condicional, mais sutil que o de `proficiencia`, que só apareceu porque fui checar a tabela cruzada em vez de aceitar a importância alta dessas colunas no modelo diagnóstico sem questionar.
- No Dia 3, antes de aceitar o IDEB como variável-alvo, verifiquei se ele já incorporava informação de infraestrutura (o que vazaria informação das próprias features que pretendo usar) — recalculando a fórmula oficial em quase 800 mil linhas reais e confirmando que a diferença residual é só arredondamento, não um fator escondido.

Esse padrão não é coincidência: qualquer variável derivada, calculada ou agregada a partir de outra fonte tem potencial de carregar, escondido, um pedaço da resposta que estou tentando prever — ou de outra variável que planejo usar como feature. Tratar isso como um cuidado recorrente, e não como uma checklist de uma etapa só, foi o que evitou, pelo menos até aqui, treinar um modelo que parece bom no papel mas não aprendeu nada de real.

### Por que escola, e não aluno: a granularidade mais acionável

Por fim, o argumento que fecha esse reframing, e que quero deixar bem explícito: a decisão de modelar no grão de **escola**, e não de aluno, não foi só uma questão de "onde tem mais dado disponível" — foi também uma questão de utilidade prática do resultado. Um modelo que aponta "o aluno João corre risco de não ser alfabetizado" é difícil de transformar em ação: o que exatamente um gestor público faz com essa informação, de forma sistemática, para milhões de alunos? Já um modelo que aponta "a Escola Municipal X tem risco elevado de ficar abaixo da meta do IDEB" aponta direto para um plano de ação concreto e executável: investir em determinado tipo de infraestrutura, reforçar corpo docente, redirecionar recursos, acompanhar de perto. A escola é a unidade sobre a qual a gestão pública de fato consegue agir de forma direta — mais operacionalizável que o aluno individual, e mais granular (menos "borrada") que o município inteiro, que mistura escolas muito diferentes entre si dentro do mesmo território.

Ou seja: a escolha da granularidade final não foi guiada só pela disponibilidade de dado (embora isso tenha sido decisivo pra viabilidade técnica) — foi guiada também pela pergunta "que decisão esse modelo vai ajudar alguém a tomar, e essa pessoa consegue agir na unidade que eu estou prevendo?". Esse é, resumindo, o processo de reframing deste projeto: da leitura literal do enunciado (aluno), passando pela constatação de que a informação disponível estava agregada demais pra sustentar isso bem (município), até chegar numa formulação que é, ao mesmo tempo, tecnicamente viável (dado real, alvo real, crosswalk validado) e mais útil pra quem vai usar o resultado (escola).
