# Fase 4 — Planejamento do curso e das aulas

> Gerado a partir dos materiais da pasta (slides `.pptx`, scripts `.py`, notebooks e READMEs). Não existia um arquivo oficial de planejamento; os títulos vêm dos slides, então servem como ementa de consulta.

## Visão geral

| Tema | Aulas | Material | Onde executar |
|---|---|---|---|
| Deep Learning (Prof. Flavio Padilha) | 8 + desafio NovaCommerce (60% da nota) | slides `.pptx`, notebooks, READMEs | `Deep Learning/aula-0X-*/handson/hands-on.ipynb` |
| IA Generativa | 6 | slides `.pptx`, notebooks | `IA Generativa/aula0X-*/handson/*.ipynb` |
| Processamento de Linguagem Natural | 6 | scripts por slide | `PLN_completo.ipynb` (scripts originais em `aula_0X/scripts_py/`) |
| LangChain para aplicações de LLM | 5 (3 vídeos cada) | `main.py` por vídeo | `LangChain_completo.ipynb` |

Instalação: `pip install -r requirements.txt` (nesta pasta), depois o modelo do spaCy e os dados do NLTK (veja o topo do `PLN_completo.ipynb`).

## Deep Learning

Ordem sugerida: fundamentos (1–2) → visão (3–4) → sequências (5–6) → treino e produção (7–8). O desafio integra as quatro frentes; vale pensar nele a partir da aula 4.

### aula-01-fundamentos

Notebook: `Deep Learning/aula-01-fundamentos/handson/hands-on.ipynb`

1. REDES NEURAIS E DEEP LEARNING
2. PARTE 1  ·  30 min
3. Para onde estamos indo
4. Objetivos da aula
5. Contexto: o neurônio biológico
6. PARTE 2  ·  20 min
7. O neurônio artificial: formalização
8. Perceptron de Rosenblatt (1958)
9. Geometria: o que o perceptron faz
10. Hipótese estatística do modelo linear
11. Funções de ativação: por que importam?
12. PARTE 3  ·  20 min
13. ReLU: por que virou padrão
14. GELU: a ativação moderna
15. O caso XOR: a crise de 1969
16. Inicialização de pesos importa
17. Notebook-first: como testamos a hipótese
18. PARTE 4  ·  20 min
19. Exemplo didático: perceptron em NumPy
20. Exemplo didático: perceptron em PyTorch
21. Caso de negócio: Nubank
22. O que vem aí (Aula 2)
23. Referências da aula

### aula-02-mlp-backpropagation

Notebook: `Deep Learning/aula-02-mlp-backpropagation/handson/hands-on.ipynb`

1. REDES NEURAIS E DEEP LEARNING
2. PARTE 1  ·  30 min
3. Onde estamos
4. O XOR vencido: duas retas bastam
5. MLP: definição formal
6. PARTE 2  ·  20 min
7. Teorema da Aproximação Universal
8. O que o teorema não garante
9. Loss: a função objetivo
10. Descida do gradiente: como a rede aprende
11. Backpropagation: a ideia
12. PARTE 3  ·  20 min
13. Backprop passo a passo: rede de 2 camadas
14. Autograd: o milagre do PyTorch
15. Vanishing gradient: o gradiente que some
16. Notebook-first: como testamos backpropagation
17. Hands-on: MLP em PyTorch
18. PARTE 4  ·  20 min
19. Laço de treino canônico
20. Overfitting: o segundo inimigo
21. Caso de negócio: churn em telecom
22. O que vem aí: Aula 3
23. Referências

### aula-03-cnn-fundamentos

Notebook: `Deep Learning/aula-03-cnn-fundamentos/handson/hands-on.ipynb`

1. REDES NEURAIS E DEEP LEARNING
2. PARTE 1  ·  30 min
3. Onde estamos: da rede densa à convolução
4. Por que o MLP falha: imagens não são vetores
5. A operação de convolução: a definição
6. Intuição: filtros como detectores
7. PARTE 2  ·  20 min
8. Stride, padding, dilation: o tamanho da saída
9. Pooling: reduzir dimensionalidade
10. Convolução em PyTorch: a conta de parâmetros
11. Receptive field: o quanto cada neurônio enxerga
12. O viés indutivo: o que a convolução assume
13. PARTE 3  ·  20 min
14. LeNet-5 (1998): o primeiro sucesso
15. AlexNet (2012): a virada
16. VGG e Inception: uniformidade × escala
17. Convolução 1×1: um truque poderoso
18. Notebook-first: como testamos o viés convolucional
19. PARTE 4  ·  20 min
20. Hands-on: a SmallCNN dos três cenários
21. Feature maps: o que a rede aprendeu
22. Caso de negócio: inspeção visual na Embraer
23. O que vem aí: Aula 4
24. Referências: o caminho até aqui

### aula-04-cnn-avancado

Notebook: `Deep Learning/aula-04-cnn-avancado/handson/hands-on.ipynb`

1. REDES NEURAIS E DEEP LEARNING
2. PARTE 1  ·  30 min
3. Onde estamos: da CNN clássica à prática industrial
4. O problema: mais camadas, pior desempenho
5. A ideia residual: aprender o resíduo, não a função
6. PARTE 2  ·  20 min
7. Por que funciona: identidade fácil, gradiente inteiro
8. Famílias pós-ResNet: somar, concatenar, escalar
9. Transfer learning: a prática dominante
10. Duas estratégias: congelar ou destravar
11. Transfer learning: não é gratuito
12. PARTE 3  ·  20 min
13. ResNet em PyTorch: três linhas
14. Augmentation moderno: torchvision.transforms.v2
15. Explicabilidade: a matemática do Grad-CAM
16. Grad-CAM em código: torchcam
17. PARTE 4  ·  20 min
18. Auditoria visual: quando o Grad-CAM expõe vieses
19. Caso de negócio: busca visual no varejo
20. Notebook-first: como testamos transferência
21. O que vem aí: Aula 5
22. Referências: o caminho da Aula 4

### aula-05-rnn-lstm

Notebook: `Deep Learning/aula-05-rnn-lstm/handson/hands-on.ipynb`

1. REDES NEURAIS E DEEP LEARNING
2. PARTE 1  ·  30 min
3. Onde estamos: a ordem passa a importar
4. O problema: sequências não cabem em vetor fixo
5. A RNN vanilla: um estado que atravessa o tempo
6. Unrolling: uma rede profunda no eixo do tempo
7. PARTE 2  ·  20 min
8. BPTT: o vanishing gradient temporal
9. LSTM (1997): três válvulas sobre a memória
10. Cell state: a esteira que atravessa o tempo
11. Por que a LSTM não some: soma em vez de produto
12. GRU (2014): a LSTM enxuta
13. PARTE 3  ·  20 min
14. BiLSTM: ler a sequência nos dois sentidos
15. Trade-offs: o que escolher em produção
16. RNN em PyTorch: três linhas e um tensor 3-D
17. Seq2Seq: comprimir e expandir sequências
18. Teacher forcing: o atalho que cobra depois
19. PARTE 4  ·  20 min
20. Caso de negócio: ETA de entrega em tempo real
21. Notebook-first: como medimos memória temporal
22. Hands-on: RNN, GRU e LSTM no forecasting diário
23. Limitações: o preço da recorrência
24. State Space Models: a alternativa de 2024
25. Referências: o caminho da Aula 5

### aula-06-attention-transformers

Notebook: `Deep Learning/aula-06-attention-transformers/handson/hands-on.ipynb`

1. REDES NEURAIS E DEEP LEARNING
2. PARTE 1  ·  30 min
3. Onde estamos: a recorrência chega ao limite
4. O gargalo do Seq2Seq: uma frase em um vetor só
5. Atenção (Bahdanau, 2015): olhar toda a entrada
6. PARTE 2  ·  20 min
7. Scaled dot-product: a atenção em uma equação
8. Self-attention: cada posição olha para as outras
9. Multi-head: várias perguntas ao mesmo tempo
10. Positional encoding: devolver a ordem à sequência
11. O bloco encoder: atenção, FFN e dois residuais
12. PARTE 3  ·  20 min
13. Encoder, decoder ou ambos: três famílias
14. Vieses arquiteturais: qual hipótese cabe no dado
15. Transformers em 2026: do BERT-base à fronteira
16. Hands-on: um bloco Transformer em PyTorch
17. Notebook-first: como testamos a autoatenção
18. PARTE 4  ·  20 min
19. Visualizando atenção: diagnóstico, não explicação
20. Limitações: o preço do custo quadrático
21. Caso de negócio: reviews no Mercado Livre
22. O que vem aí: otimização e regularização (Aula 7)
23. Referências: o caminho da Aula 6

### aula-07-otimizacao

Notebook: `Deep Learning/aula-07-otimizacao/handson/hands-on.ipynb`

1. REDES NEURAIS E DEEP LEARNING
2. PARTE 1  ·  30 min
3. Onde estamos: da arquitetura ao treino
4. Agenda: o que faz um treino convergir
5. O problema: a perda não é uma tigela
6. SGD: o passo mais simples
7. PARTE 2  ·  20 min
8. SGD com momentum: memória da direção
9. RMSprop e Adam: um passo por parâmetro
10. AdamW e Lion: o padrão de 2026
11. Regime de treino: um sistema acoplado
12. Learning rate: o hiperparâmetro nº 1
13. LR schedulers: o passo muda ao longo do treino
14. PARTE 3  ·  20 min
15. Regularização L2: penalizar pesos grandes
16. Dropout: um ensemble implícito
17. Normalização: BatchNorm, LayerNorm e GroupNorm
18. Data augmentation: mais dados sem coletar dados
19. Early stopping: saber a hora de parar
20. SAM: procurar mínimos planos
21. PARTE 4  ·  20 min
22. HPO com Optuna: busca guiada, não grade
23. Caso de negócio: Olist e a previsão de demanda
24. Notebook-first: isolando o efeito do treino
25. Checklist prático: como treinar em 2026
26. O que vem aí: Aula 8 — MLOps e deploy
27. Referências: sete anos de padrões

### aula-08-mlops-deploy-tendencias

Notebook: `Deep Learning/aula-08-mlops-deploy-tendencias/handson/hands-on.ipynb`

1. REDES NEURAIS E DEEP LEARNING
2. PARTE 1  ·  30 min
3. Onde estamos: do treino ao plantão 24/7
4. Agenda: o que separa o notebook da produção
5. MLflow Tracking: todo experimento vira registro
6. MLflow Registry: o ciclo de vida do modelo
7. PARTE 2  ·  20 min
8. Promover é decisão de risco: não é ranking de métrica
9. Export para ONNX: um grafo, muitos runtimes
10. Serving: TorchServe e Triton
11. Quantização: o mesmo modelo em 8 bits
12. Pruning e distillation: modelos menores de propósito
13. Monitoramento: o que olhar depois do deploy
14. PARTE 3  ·  20 min
15. Detecção de drift: KS, PSI e limiar de ação
16. Notebook-first: o experimento operacional da aula
17. Multimodalidade: imagem e texto no mesmo espaço
18. State Space Models: Mamba e o custo linear
19. Modelos de fronteira 2026: o mapa atual
20. PARTE 4  ·  20 min
21. RAG: gerar com a fonte na mão
22. Caso Americanas: busca multimodal em 200M SKUs
23. Ética e governança: o que não é opcional
24. Depois do curso: como continuar sozinho
25. Fechamento: do perceptron ao plantão 24/7
26. Referências: os papers que sustentam a aula

## IA Generativa

### Aula 01

Notebook: `IA Generativa/aula01-transformer/handson/aula01_handson.ipynb`

1. IA GENERATIVA
2. IA GENERATIVA
3. Os Transformers se tornaram a base dos modelos fundacionais modernos de linguagem, visão, áudio e multimodalid
4. Diferenciar limitações de RNNs, CNNs e a solução proposta pela atenção.   Descrever a arquitetura encoder-deco
5. Redes recorrentes processam sequências passo a passo: cada estado oculto depende do estado anterior e da entra
6. CNNs permitem maior paralelismo que RNNs, mas capturam relações locais por janelas ou filtros.   Para relacion
7. O Transformer permite que cada posição da sequência consulte diretamente outras posições relevantes.   A atenç
8. O Transformer original foi proposto como uma arquitetura composta por encoder e decoder.   O encoder transform
9. Cada camada do encoder combina multi-head self-attention e uma rede feed-forward posicional.   Cada camada do 
10. IA GENERATIVA
11. Modelos Transformer não processam palavras diretamente; eles processam tokens representados por números.   A t
12. O BPE começa tratando caracteres como tokens e junta pares frequentes de forma iterativa.   O processo cria um
13. Depois da tokenização, cada token é associado a um vetor de alta dimensão chamado embedding.   Embeddings perm
14. Cada embedding é projetado em três representações: Query (Q), Key (K) e Value (V).   A Query de um token é com
15. A atenção calcula similaridade entre consultas e chaves por produto escalar.   O resultado é escalado pela rai
16. Em vez de uma única atenção, o Transformer usa múltiplas cabeças em paralelo.   Cada cabeça aprende projeções 
17. Como o Transformer remove recorrência e convolução, ele precisa injetar informação de posição explicitamente. 
18. IA GENERATIVA
19. Pré-treinamento constrói a base do modelo com grandes volumes de dados e objetivos auto supervisionados.   Fin
20. LoRA congela os pesos originais do modelo e injeta matrizes treináveis de baixa dimensão em camadas do Transfo
21. Encoder-only: usa apenas o codificador, com atenção bidirecional, ideal para tarefas de compreensão. Exemplo: 
22. Na atenção bidirecional, cada token pode observar tokens anteriores e posteriores.   Essa estratégia é útil pa
23. Inspecionar a tokenização de frases curtas usando uma biblioteca de modelos Transformer.   Executar inferência
24. Transformers deixaram de ser apenas uma inovação acadêmica e se tornaram infraestrutura de produto em assisten
25. Você compreendeu por que RNNs e CNNs tinham limitações para dependências longas e treinamento paralelo.   Estu
26. VASWANI, Ashish et al. Attention is all you need. In: Advances in Neural Information Processing Systems 30. [S
27. (sem texto)

### Aula 02

Notebook: `IA Generativa/aula02-rag/handson/aula02_handson.ipynb`

1. IA GENERATIVA
2. IA GENERATIVA
3. Apresentar o objetivo da aula: compreender como um LLM pode responder usando evidências externas, atualizadas 
4. LLMs geram texto com alta fluência, mas o conhecimento interno depende dos dados e da data de treinamento.   Q
5. Retrieval-Augmented Generation combina recuperação de informações com geração de texto.   Antes de responder, 
6. Memória paramétrica: conhecimento comprimido nos pesos do modelo durante o treinamento.   Memória não paramétr
7. Quando a solução precisa responder sobre documentos internos, normas, contratos, manuais, políticas ou bases t
8. Benefícios: reduz alucinações, melhora atualização do conhecimento, permite citações e facilita governança.   
9. Entender os fundamentos conceituais do RAG.   Descrever o pipeline completo: ingestão, chunking, embeddings, i
10. IA GENERATIVA
11. O pipeline começa com documentos brutos e termina com respostas fundamentadas em evidências.   As etapas princ
12. A ingestão seleciona os arquivos que farão parte da base de conhecimento.   Formatos comuns incluem PDF, HTML,
13. Parsing extrai texto estruturado de documentos digitais.   OCR converte imagens ou PDFs escaneados em texto pe
14. Documentos longos raramente cabem inteiros na janela de contexto do LLM.   Dividir em chunks permite recuperar
15. Fixed-size chunking: divide por quantidade de caracteres ou tokens, geralmente com overlap.   Recursive chunki
16. Modelos de IA operam sobre representações numéricas; embeddings transformam texto em vetores.   Textos semanti
17. Depois dos embeddings, os chunks são enviados para um índice que permite busca rápida e escalável.   Cada item
18. Quando a pergunta chega, ela também é transformada em consulta textual e/ou vetorial.   O mecanismo de busca r
19. IA GENERATIVA
20. A recuperação lexical busca correspondências de palavras-chave entre a pergunta e os documentos.   É forte qua
21. A recuperação vetorial busca proximidade semântica, mesmo quando as palavras da pergunta não aparecem literalm
22. Em bases pequenas, é possível comparar a consulta com todos os vetores.   Em bases grandes, a busca exata fica
23. Na prática, a melhor estratégia costuma combinar busca lexical e vetorial.   A busca lexical preserva precisão
24. O re-ranking reorganiza os candidatos recuperados para colocar os trechos mais úteis no topo.   Filtros por me
25. O LLM recebe os trechos recuperados como contexto de grounding.   O prompt deve orientar o modelo a responder 
26. Recall@k mede quantos itens relevantes aparecem entre os primeiros k resultados recuperados.   Se a evidência 
27. Faithfulness ou groundedness mede se a resposta está sustentada pelos trechos recuperados.   Answer relevancy 
28. Além de métricas de qualidade, é preciso monitorar latência, custo por requisição, consumo de tokens, erros e 
29. IA GENERATIVA
30. Em produção, o RAG precisa separar ingestão, armazenamento, indexação, recuperação, geração e monitoramento.  
31. Amazon Bedrock Knowledge Bases permite conectar modelos fundacionais a bases próprias para RAG.   IAM, KMS, Gu
32. Azure AI Search suporta busca textual, vetorial e híbrida para fundamentar respostas de aplicações GenAI.   A 
33. Vertex AI RAG Engine facilita a construção de aplicações de LLM aumentadas por contexto.   Vertex AI Text Embe
34. O RAG se tornou uma camada estrutural de produtos corporativos de IA generativa.   A disputa competitiva está 
35. Selecionar uma base documental pequena e preparar os arquivos para ingestão.   Criar chunks, gerar embeddings 
36. RAG combina memória paramétrica do LLM com memória não paramétrica em bases externas.   O pipeline passa por i
37. LEWIS, P. et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Informat
38. MALKOV, Y.; YASHUNIN, D. Efficient and robust approximate nearest neighbor search using Hierarchical Navigable
39. (sem texto)

### Aula 03

Notebook: `IA Generativa/aula03-prompt/handson/aula03_handson.ipynb`

1. IA GENERATIVA
2. IA GENERATIVA
3. Apresentar a Engenharia de Prompt como a disciplina que transforma interação com LLMs em processo sistemático,
4. LLMs operam de forma probabilística e condicionada ao contexto textual recebido.  O prompt é a principal inter
5. Diferentemente de programas tradicionais, modelos de linguagem não executam instruções de modo determinístico.
6. Instrução: define a tarefa e o objetivo da resposta.  Contexto: fornece dados, restrições, público-alvo e info
7. Um prompt pode ser tratado como um contrato entre sistema e modelo.  Ele declara comportamento esperado, forma
8. Modelos de linguagem processam tokens, não necessariamente palavras completas.  A tokenização impacta custo, l
9. Prompts robustos separam claramente instruções, dados, exemplos e saída esperada.  Delimitadores, tags XML e b
10. Compreender o papel do prompt como interface de controle de LLMs.  Reconhecer os componentes de um prompt estr
11. IA GENERATIVA
12. No zero-shot, o modelo recebe apenas a instrução da tarefa, sem exemplos.  É simples, rápido e econômico, mas 
13. No one-shot, um exemplo orienta o modelo; no few-shot, vários exemplos definem padrão, estilo e formato.  Os e
14. Chain-of-Thought incentiva o modelo a decompor problemas complexos em etapas intermediárias.  A técnica é útil
15. Self-consistency gera múltiplas trajetórias de solução e seleciona a resposta mais consistente ou frequente.  
16. Prompt chaining divide uma tarefa complexa em etapas menores e conectadas.  A saída de uma etapa serve como en
17. A definição de papel orienta vocabulário, tom, profundidade e critérios de decisão.  Personas úteis são especí
18. Saídas estruturadas facilitam integração com sistemas, validação automática e testes.  Schemas reduzem fuga de
19. Começar com um prompt zero-shot simples.  Adicionar contexto, exemplos few-shot, formato de saída e critérios 
20. IA GENERATIVA
21. A temperatura ajusta a aleatoriedade da geração.  Valores baixos tornam a resposta mais conservadora e previsí
22. Top-k limita a escolha aos k tokens mais prováveis.  Top-p seleciona o conjunto mínimo de tokens cuja probabil
23. Alucinação: geração de informação incorreta com aparência de confiança.  Ambiguidade: instruções abertas demai
24. Prompt injection ocorre quando uma instrução maliciosa tenta sobrescrever regras do sistema.  No ataque indire
25. Faithfulness mede se a resposta está sustentada pelos dados fornecidos.  Contextual precision e recall medem a
26. Prompts devem ter versão, dono, objetivo, dataset de teste e histórico de alterações.  Observabilidade monitor
27. Empresas usam prompts estruturados para atendimento, copilotos internos, automação de documentos e suporte téc
28. Frameworks como LangChain e LlamaIndex ajudam a organizar prompts, chains, ferramentas e recuperação de contex
29. Especificar objetivo, público, escopo, dados de entrada, restrições e formato de saída.  Separar instruções de
30. Engenharia de Prompt como disciplina estruturada para controlar LLMs.  Técnicas zero-shot, few-shot, Chain-of-
31. Desafio: escolher uma tarefa real e construir um prompt robusto com instrução, contexto, exemplos e formato de
32. BROWN, T. et al. Language Models are Few-Shot Learners. In: Advances in Neural Information Processing Systems 
33. ZHOU, K. et al. Large Language Model Prompt Engineering: A Survey. [S.l.: s.n.], 2023.  LIU, N. F. et al. Lost
34. (sem texto)

### Aula 04

Notebook: `IA Generativa/aula04-agentes/handson/aula04_handson.ipynb`

1. IA GENERATIVA
2. IA GENERATIVA
3. Aplicações iniciais de IA generativa operavam como fluxos lineares: pergunta, contexto, resposta e encerrament
4. Diferenciar agentes, workflows e pipelines determinísticos.  Explicar o paradigma ReAct e o papel de observaçõ
5. Um agente observa um ambiente, mantém estado interno, escolhe uma ação e avalia o resultado para decidir o pró
6. Modelo de linguagem: interpreta objetivos, planeja respostas e seleciona ações.  Orquestrador: controla o loop
7. Workflow: o controle está no código; as etapas são definidas previamente pelo desenvolvedor.  Agente: parte do
8. A flexibilidade aumenta a utilidade, mas reduz previsibilidade e testabilidade.  Riscos comuns: loops, chamada
9. ReAct combina raciocínio e ação em ciclos: o agente interpreta o objetivo, decide uma ação, recebe uma observa
10. Objetivo: implementar um loop simples com pergunta do usuário, decisão do modelo, chamada de ferramenta e resp
11. IA GENERATIVA
12. Tool calling é o mecanismo pelo qual o modelo solicita a execução de uma ferramenta externa.  O modelo não exe
13. Uma ferramenta bem definida deve ter nome, descrição, schema de entrada, schema de saída e classificação de ri
14. Aplicar princípio do menor privilégio: cada ferramenta deve acessar apenas o necessário.  Separar ferramentas 
15. Protocolos buscam padronizar como agentes acessam ferramentas, recursos e outros agentes.  MCP expõe capacidad
16. Protocolos reduzem acoplamento e facilitam reutilização, versionamento e autorização.  Mesmo com protocolo com
17. ReAct decide incrementalmente, mas tarefas complexas podem exigir separação entre planejamento e execução.  Um
18. Plan-and-Solve separa construção do plano e resolução da tarefa.  Tree of Thoughts explora múltiplos caminhos 
19. Adicionar schema de entrada para a ferramenta do agente.  Validar argumentos antes da execução.  Adicionar lim
20. IA GENERATIVA
21. Em arquiteturas hierárquicas, um agente supervisor coordena agentes especializados.  O supervisor recebe a tar
22. Agentes colaboradores podem trabalhar em tarefas complexas e multi-etapa sob coordenação de um supervisor.  O 
23. Memória de curto prazo: contexto da sessão atual, mensagens recentes, ações executadas e estado temporário.  M
24. Em produção, ferramentas viram APIs autenticadas, microserviços, conectores MCP, funções serverless ou serviço
25. Monitorar agentes exige mais do que latência, taxa de erro e throughput.  É necessário acompanhar qual agente 
26. A avaliação verifica se o agente agiu de forma correta, segura, útil, eficiente e rastreável.  Avaliar apenas 
27. O mercado de agentes avança em três frentes: automação corporativa, padronização de ferramentas e observabilid
28. Agentes são sistemas de decisão compostos por objetivo, estado, observação, ação, ferramentas e memória.  Work
29. Projetar um agente para um cenário corporativo simples.  Definir objetivo, ferramentas, schemas, riscos, memór
30. ANTHROPIC. Building effective agents. Anthropic Engineering, 2024.  YAO, Shunyu et al. ReAct: Synergizing Reas
31. OPENAI. Introducing Structured Outputs in the API. OpenAI, 2024.  OPENAI. Tracing — OpenAI Agents SDK. OpenAI 
32. (sem texto)

### Aula 05

Notebook: `IA Generativa/aula05-multimodal/handson/aula05_handson.ipynb`

1. IA GENERATIVA
2. IA GENERATIVA
3. Durante muito tempo, modelos de IA foram especializados: visão para imagens, NLP para texto e fala para áudio.
4. Definir multimodalidade no contexto de modelos fundacionais.  Explicar como embeddings textuais e visuais pode
5. Multimodalidade é a capacidade de processar e relacionar múltiplos tipos de dados, como texto, imagem, áudio, 
6. Grande parte do conhecimento corporativo aparece em gráficos, fluxogramas, prints de tela, plantas, exames, fo
7. Uma representação traduz um objeto complexo para um vetor numérico.  Texto pode ser convertido em embedding; i
8. O CLIP alinha imagens e textos por meio de aprendizado contrastivo.  O modelo recebe lotes com imagens e descr
9. O modelo não aprende apenas pixels ou palavras isoladas; ele aprende relações entre padrões visuais e padrões 
10. Uma arquitetura comum combina um encoder visual com um modelo de linguagem.  O encoder visual transforma a ima
11. Treinar todos os componentes do zero é caro e exige grandes bases multimodais.  Estratégias modernas reutiliza
12. A fusão entre modalidades pode ocorrer por concatenação de tokens visuais e textuais ou por mecanismos de aten
13. Vision-Language Models lidam com tarefas que combinam visão e linguagem.  Exemplos: image captioning, visual q
14. IA GENERATIVA
15. O instruction tuning multimodal ajusta o modelo para seguir instruções humanas envolvendo imagens e texto.  Em
16. No RAG textual, documentos são divididos em chunks, vetorizados, indexados e recuperados para fundamentar resp
17. Em sistemas acadêmicos e corporativos, não basta responder: é necessário indicar de onde veio a resposta.  Em 
18. O chunking multimodal exige mais cuidado que o chunking textual.  Uma tabela não deve ser separada de seu títu
19. Um chunk enriquecido pode conter texto da página, legenda da figura, descrição visual gerada por VLM e OCR de 
20. Avaliar modelos multimodais é mais complexo do que avaliar modelos textuais.  Além de exatidão, relevância e f
21. DocVQA avalia pergunta e resposta sobre imagens de documentos.  MMMU avalia compreensão e raciocínio multimoda
22. Custo: imagens, PDFs e vídeos podem consumir mais recursos que texto puro.  Latência: pipelines com OCR, descr
23. IA GENERATIVA
24. Em produção, muitas organizações preferem serviços gerenciados para acelerar o uso de modelos multimodais.  Es
25. No Azure, a estratégia multimodal pode combinar Azure AI Foundry, Azure OpenAI, Azure AI Search e serviços de 
26. No Google Cloud, o Vertex AI disponibiliza modelos Gemini para processamento de texto, imagem, vídeo e documen
27. Entrada: imagem ou PDF com texto e elementos visuais.  Etapa 1: envio para um modelo multimodal para gerar des
28. Preservar a relação entre texto, figura, legenda, tabela e página.  Armazenar metadados suficientes para audit
29. LLMs multimodais ampliam o uso de IA generativa em documentos corporativos, suporte técnico, saúde, educação, 
30. Como LLMs multimodais conectam texto, imagem e documentos por meio de representações conjuntas e aprendizado c
31. RADFORD, Alec et al. Learning Transferable Visual Models From Natural Language Supervision. arXiv, 2021.  LI, 
32. (sem texto)

### Aula 06

Notebook: `IA Generativa/aula06-etica-seguranca/handson/aula06_handson.ipynb`

1. IA GENERATIVA
2. IA GENERATIVA
3. A IA generativa passou a compor fluxos reais de decisão, atendimento, educação, saúde, finanças, desenvolvimen
4. Explicar GenAI como sistema sociotécnico e não apenas como modelo estatístico.  Classificar riscos em famílias
5. Um assistente corporativo não é apenas uma chamada a um LLM.  Ele envolve documentos internos, bases vetoriais
6. Quando um modelo inventa uma referência, temos uma falha de factualidade.  Quando essa falha ocorre em context
7. O NIST AI Risk Management Framework apoia organizações no gerenciamento de riscos em concepção, desenvolviment
8. Govern: estabelecer papéis, políticas, responsabilidades e cultura de gestão de risco.  Map: compreender conte
9. Viés, discriminação e injustiça algorítmica.  Toxicidade, conteúdo abusivo e dano psicológico.  Integridade in
10. IA GENERATIVA
11. Modelos treinados em grandes corpus podem reproduzir padrões históricos de desigualdade e estereótipos.  Viés 
12. Sistemas de GenAI podem gerar insultos, discurso de ódio, conteúdo inadequado, incentivo à violência ou instru
13. Confabulação ocorre quando o modelo produz respostas plausíveis, mas falsas ou sem sustentação.  Em sistemas c
14. Dados pessoais podem entrar por prompts, anexos, documentos RAG, logs, histórico, fine-tuning, telemetria e fe
15. Prompt injection ocorre quando uma entrada textual, visual ou multimodal altera o comportamento do modelo de f
16. Separar instruções confiáveis de dados externos não confiáveis.  Marcar conteúdo recuperado como evidência, e 
17. Guardrails são mecanismos técnicos e organizacionais para manter o comportamento do sistema dentro de limites 
18. IA GENERATIVA
19. Governança começa antes do prompt: é preciso saber quais dados entram, de onde vêm, quem autorizou, por quanto
20. Incidentes podem envolver vazamento de dados, conteúdo discriminatório, obediência a documento malicioso, exec
21. A conformidade em GenAI exige inventário, classificação de risco, uso pretendido, documentação técnica e evidê
22. O mercado exige cada vez mais sistemas de GenAI auditáveis, seguros e alinhados a políticas corporativas.  A t
23. Transformar políticas de uso em verificações automatizadas.  Detectar informações pessoais identificáveis em e
24. Escopo e uso pretendido documentados.  Dados classificados, minimizados e com permissões preservadas.  Riscos 
25. Ética em GenAI não é apenas discussão filosófica; aparece em dados, métricas, arquitetura, permissões, documen
26. TABASSI, E. Artificial Intelligence Risk Management Framework (AI RMF 1.0). Gaithersburg: National Institute o
27. GRESHAKE, K. et al. Not what you’ve signed up for: Compromising Real-World LLM-Integrated Applications with In
28. (sem texto)

## Processamento de Linguagem Natural

Os títulos abaixo são dos slides que têm código (os demais são teoria e não geraram script).

### Aula 01

- Slide 30: Polissemia em código
- Slide 49: Lei de zipf a distribuição estatística dos tokens em qualquer corpus
- Slide 52: Setup do ambiente bibliotecas para começar
- Slide 53: Passo 1 — carregando o corpus com nltk tokenização e contagem
- Slide 54: Passo 2 — análise linguística com spacy o padrão da indústria
- Slide 55: Spacy: named entity recognition identificando pessoas, lugares e organizações
- Slide 56: Passo 3 — primeira vetorização do texto pré-processado a vetores numéricos

### Aula 02

- Slide 13: Regex em python
- Slide 14: Setup do ambiente
- Slide 15: Limpeza: html, urls e contrações
- Slide 18: Tratamento de acentos decisão sensível ao downstream
- Slide 20: Pipeline completo: limpar_normalizar() a função que costura tudo
- Slide 30: Família 1: split() ingênuo o baseline didático
- Slide 31: Família 2a: nltk word_tokenize robusto, com suporte a português
- Slide 32: Família 2b: spacy o padrão da indústria moderna
- Slide 33: Sentence tokenization quebrando texto em frases
- Slide 37: Bpe: byte-pair encoding o algoritmo dos transformers
- Slide 40: Curiosidade por que gpt e bert tokenizam de forma estranha
- Slide 50: Rslp em ação o stemmer brasileiro
- Slide 51: Lemmatization com spacy mais lento, mais preciso
- Slide 54: O que são stop words? palavras frequentes com pouco conteúdo semântico
- Slide 57: Pipeline nlp final tudo costurado em uma função

### Aula 03

- Slide 15: Bow manualmente em python sem libs, para entender por dentro
- Slide 16: Bow com scikit-learn countvectorizer faz tudo
- Slide 17: Saída típica do countvectorizer matriz documento-termo em ação
- Slide 18: Esparsidade matriz cheia de zeros
- Slide 19: Scipy.sparse
- Slide 31: Trigrams (n=3)
- Slide 32: Range (1, 2): unigrams + bigrams
- Slide 33: Implementação prática
- Slide 35: Filtros: min_df e max_df
- Slide 38: Bigrams + sentiment analysis em pt-br
- Slide 48: Tf: term frequency
- Slide 49: Idf: inverse document frequency
- Slide 51: Tf-idf manualmente
- Slide 52: Saída do tf-idf manual
- Slide 53: Tfidfvectorizer em produção
- Slide 55: Bow vs tf-idf: tarefa real

### Aula 04

- Slide 8: Embeddings densos poucos números, muita semântica
- Slide 17: Treinando word2vec com gensim setup e treinamento em pt-br
- Slide 18: Usando o modelo treinado acessando vetores e similaridades
- Slide 31: Fasttext resolve oov não há mais "palavra desconhecida"
- Slide 36: Fasttext oficial (facebook) 157 línguas, incluindo pt
- Slide 38: Quando vale fine-tuning? adaptando pré-treinados ao seu domínio
- Slide 47: Carregando embeddings pré-treinados fasttext oficial em pt-br
- Slide 49: Similaridade do cosseno em código com gensim
- Slide 51: Analogias em código "rei está para homem como x está para mulher"
- Slide 56: Visualização em código clusters semânticos com t-sne

### Aula 05

- Slide 12: Suavização de laplace evita probabilidades zero
- Slide 14: Multinomial nb + tf-idf implementação em pipeline
- Slide 15: Saída do classification_report métricas por classe
- Slide 17: Regressão logística o cavalo-de-batalha da classificação
- Slide 18: Interpretabilidade dos coeficientes a grande vantagem da lr
- Slide 29: Class_weight='balanced' lidando com classes desbalanceadas
- Slide 32: Pipeline em código vetorização + classificador em um objeto
- Slide 33: Gridsearchcv no pipeline tuning de vetorização + classificador juntos
- Slide 34: Validação cruzada estratificada preservando proporções de classe
- Slide 40: Classification_report métricas completas em uma chamada
- Slide 49: Bio tagging o esquema padrão para anotação de ner
- Slide 50: Ner com spacy em pt-br pt_core_news_lg pré-treinado
- Slide 51: Saída do ner em pt-br entidades reconhecidas pelo spacy
- Slide 52: Avaliação de ner por entidade, não por token
- Slide 56: Pipeline completo tf-idf + logreg + ner em paralelo
- Slide 57: Função enriquecer_review sentimento + entidades em uma chamada

### Aula 06

- Slide 19: Lda em gensim — código completo reviews em pt-br, do bow ao coherence
- Slide 38: Pyldavis — o padrão de visualização a ferramenta que vendeu lda a cientistas de dados
- Slide 53: Bertopic em pt-br — código completo setup com sentence transformer multilingual
- Slide 54: Bertopic — visualizações nativas inspiradas no pyldavis e mais ricas ainda
- Slide 58: Bertopic vs lda — comparação direta código de benchmark em corpus de reviews pt-br

## LangChain para aplicações de LLM

Resumo inferido do código dos `main.py` (não há ementa no material).

### Aula 01

- Vídeo 1: Chain LCEL básica com Gemini (prompt | modelo | parser)
- Vídeo 2: Ferramentas (cálculo de custo de tokens) e agente
- Vídeo 3: Agente com busca na web (Tavily)

### Aula 02

- Vídeo 1: Limpeza de texto e chains
- Vídeo 2: Roteamento de consultas
- Vídeo 3: Agente com ferramentas e dois provedores (OpenAI/Gemini)

### Aula 03

- Vídeo 1: Embeddings e busca vetorial (FAISS)
- Vídeo 2: Multi-query retrieval
- Vídeo 3: Memória de chat em SQLite

### Aula 04

- Vídeo 1: Ferramenta `consultar_vendas` com schema
- Vídeo 2: Agente que usa a ferramenta
- Vídeo 3: API FastAPI agêntica com autenticação por chave

### Aula 05

- Vídeo 1: LangGraph: LCEL e grafo de estados
- Vídeo 2: Checkpoints, edição de estado e histórico
- Vídeo 3: Supervisor e execução sequencial vs paralela
