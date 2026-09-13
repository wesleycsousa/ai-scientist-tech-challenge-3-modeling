# Tech Challenge Fase 3 — Predição de Desempenho Escolar (IDEB)

Repositório do Tech Challenge da Fase 3 da pós-graduação em Ciência de Dados. O objetivo é construir um modelo supervisionado de classificação binária para prever se uma escola atinge um patamar de desempenho educacional considerado adequado, usando variáveis de infraestrutura, corpo docente, matrícula e contexto territorial.

O alvo do modelo é `alvo_ideb`: `1` se o IDEB (Índice de Desenvolvimento da Educação Básica, calculado pelo INEP) da escola for maior ou igual a 6,0, e `0` caso contrário. O corte de 6,0 é a meta nacional de referência do MEC/INEP. A granularidade do problema é por escola.

O projeto começou com uma proposta diferente: prever alfabetização por aluno, reaproveitando a camada Gold da Fase 2 (Engenharia de Dados). Ao longo da exploração percebi que essa camada, agregada por município, não tinha o nível de detalhe que o desafio pede, e que a base de alunos não tinha um crosswalk confiável com os dados de infraestrutura escolar que eu queria usar como feature. Pesquisando uma fonte externa de desempenho por escola encontrei o IDEB, que resolveu o problema no grão certo. A reformulação completa, com toda a investigação até chegar nessa decisão, está documentada em `notebooks/eda/README.md` e em `reports/decisoes.md` (seção "Reflexão").

A base final vem do Censo Escolar (tabela `escola_completo`, camada Silver herdada da Fase 2), cruzada com o IDEB por escola via `id_escola` (crosswalk validado, ~95,9% de interseção entre as duas fontes), filtrada para rede pública, anos iniciais do ensino fundamental, ano de referência 2025.

> Este README ainda está em construção — as seções de resultados, interpretação e aplicação prática vão ser completadas conforme a etapa de modelagem avança.

## Estrutura do repositório

```
├── data/                  # dados locais (não versionados, vêm do S3)
├── notebooks/
│   ├── eda/               # exploração e reformulação do problema (00 a 08)
│   └── modelagem/         # baseline e otimização de modelos
├── src/
│   ├── preprocessing/     # leitura dos dados e pipeline de pré-processamento
│   ├── modeling/          # treinamento dos modelos
│   ├── evaluation/        # métricas e validação
│   └── visualization/     # gráficos e visualizações
├── reports/               # decisões tomadas, insights, achados
├── images/                # gráficos exportados
├── requirements.txt
└── .env.example           # modelo de variáveis de ambiente (credenciais AWS)
```

## Como rodar

1. Criar e ativar um ambiente virtual:
   ```
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   ```
2. Instalar as dependências:
   ```
   pip install -r requirements.txt
   ```
3. Copiar `.env.example` para `.env` e preencher com as credenciais da AWS.
4. Rodar os notebooks a partir de `notebooks/eda/`, em ordem numérica, seguidos dos notebooks em `notebooks/modelagem/`. A base de modelagem também pode ser gerada diretamente rodando `src/preprocessing/build_base_escola.py`.
