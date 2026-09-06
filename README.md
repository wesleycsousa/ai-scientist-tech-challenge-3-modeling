# Tech Challenge Fase 3 — Predição de Alfabetização no Brasil

Repositório do Tech Challenge da Fase 3 da pós-graduação em Ciência de Dados. O objetivo é construir um modelo supervisionado de classificação binária para prever se um aluno será considerado alfabetizado ou não, usando variáveis educacionais, territoriais e socioeconômicas.

Este repositório dá continuidade ao trabalho feito na Fase 2 (Engenharia de Dados), reaproveitando os dados tratados nas camadas Silver e Gold daquele projeto.

> Este README ainda está em construção — vai ser completado ao longo do desenvolvimento com o contexto do problema, as etapas de modelagem, os resultados e os insights encontrados.

## Estrutura do repositório

```
├── data/                  # dados locais (não versionados, vêm do S3)
├── notebooks/             # notebooks de EDA, modelagem e interpretabilidade
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
4. Rodar os notebooks/scripts a partir de `src/preprocessing/`.
