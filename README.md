# Predição de Óbito por COVID-19 — Dashboard Interativo

> Trabalho Acadêmico · Disciplina: Redes Neurais  
> Classificação binária com **MLPClassifier (Scikit-learn)** sobre dataset Kaggle com 1M+ registros

---

## Equipe

| Nome | Contribuição |
|------|-------------|
| Bruno Martins | Modelagem e experimentos |
| Christian Malacize | Pipeline de dados |
| Bruno Bomfim | Análise de resultados |
| Lucas Galli | Pré-processamento |
| Raphael | Avaliação e métricas |

---

## Sobre o Projeto

O objetivo é prever se um paciente diagnosticado com COVID-19 veio a óbito (`1`) ou sobreviveu (`0`), usando um classificador de Perceptron Multicamadas (MLP). O maior desafio foi o **desbalanceamento severo de classes (13:1)**, contornado com ajuste de threshold pós-treinamento.

---

## Dataset

- **Fonte:** Kaggle — COVID-19 Patient Data (México)
- **Total de registros:** 1.048.575
- **Amostra utilizada:** 100.000 (estratificada, `random_state=42`)
- **Features de entrada:** 15 atributos (14 categóricos + 1 numérico)
- **Variável alvo:** derivada de `DATE_DIED` → `0` (sobreviveu) / `1` (óbito)

### Distribuição das Classes

```
Sobreviveu (0):  971.633  →  92.66%
Óbito      (1):   76.942  →   7.34%
Proporção:  ~13:1  (severo desbalanceamento)
```

---

## Features Utilizadas

| Feature | Tipo | Encoder |
|---------|------|---------|
| AGE | Numérica | StandardScaler |
| SEX | Categórica | OneHotEncoder |
| PNEUMONIA | Categórica | OneHotEncoder |
| DIABETES | Categórica | OneHotEncoder |
| COPD | Categórica | OneHotEncoder |
| ASTHMA | Categórica | OneHotEncoder |
| INMSUPR | Categórica | OneHotEncoder |
| HIPERTENSION | Categórica | OneHotEncoder |
| OTHER_DISEASE | Categórica | OneHotEncoder |
| CARDIOVASCULAR | Categórica | OneHotEncoder |
| OBESITY | Categórica | OneHotEncoder |
| RENAL_CHRONIC | Categórica | OneHotEncoder |
| TOBACCO | Categórica | OneHotEncoder |
| USMER | Categórica | OneHotEncoder |
| MEDICAL_UNIT | Categórica | OneHotEncoder |

**Colunas removidas:** `DATE_DIED` (originou o alvo), `INTUBED`, `ICU`, `PATIENT_TYPE`, `CLASIFFICATION_FINAL`, `PREGNANT`

---

## Pipeline

```
Dataset (Kaggle)
   ↓
Definição do Alvo  →  DATE_DIED: preenchido = óbito (1), NaN = sobreviveu (0)
   ↓
Limpeza  →  remoção de 6 colunas; 97/99 tratados como NaN; dropna()
   ↓
Amostragem Estratificada  →  100.000 registros
   ↓
Pré-processamento  →  ColumnTransformer(StandardScaler + OneHotEncoder)
   ↓
Train/Test Split  →  80% treino / 20% teste  (stratify=y)
   ↓
Treinamento MLP  →  27 configurações testadas
   ↓
Ajuste de Threshold  →  threshold < 0.5 para aumentar recall da classe 1
   ↓
Avaliação  →  Acurácia, F1-Score, Recall, Precisão, Matriz de Confusão
```

---

## Experimentos

Foram testadas **27 configurações** (3 arquiteturas × 3 tamanhos × 3 learning rates):

### Arquiteturas

| Camadas | Configurações |
|---------|--------------|
| 1 camada oculta | `(16,)` · `(32,)` · `(64,)` |
| 2 camadas ocultas | `(16,16)` · `(32,32)` · `(64,64)` |
| 3 camadas ocultas | `(16,16,16)` · `(32,32,32)` · `(64,64,64)` |

### Learning Rates

`0.0001` · `0.001` · `0.01`

### Top 5 Resultados (por F1-Score)

| Rank | Exp # | Arquitetura | LR | Acurácia | F1-Score | Recall cls.1 |
|------|-------|-------------|-----|----------|----------|--------------|
| 🥇 1° | 11 | **(16, 16)** | 0.001 | 93.25% | **0.4639** | 40.33% |
| 2° | 21 | (16, 16, 16) | 0.01 | 93.44% | 0.4583 | 38.33% |
| 3° | 22 | (32, 32, 32) | 0.0001 | 93.45% | 0.4457 | 36.40% |
| 4° | 7 | (64,) | 0.0001 | 93.38% | 0.4414 | 36.12% |
| 5° | 2 | (16,) | 0.001 | 93.34% | 0.4380 | 35.84% |

---

## Resultados Finais

### Modelo Vencedor: `(16, 16)` com `learning_rate=0.001`

#### Comparação: Sem vs. Com Ajuste de Threshold

| Métrica | Sem Ajuste | Com Ajuste | Variação |
|---------|-----------|-----------|---------|
| Acurácia | 93.25% | 91.55% | ▼ -1.7 pp |
| F1-Score | 0.464 | **0.558** | ▲ +20.3% |
| Recall cls.1 | 40.33% | **73.55%** | ▲ +82.4% |
| Precisão cls.1 | 54.58% | 44.90% | ▼ -17.7% |

#### Matriz de Confusão (com threshold ajustado)

```
                  Predito 0    Predito 1
Real 0 (Sobreviveu)  16.962       529
Real 1 (Óbito)          380     1.129
```

---

## Conclusão

> O **ajuste de threshold** mostrou-se mais eficaz do que aumentar a complexidade da arquitetura.
> Reduzir o limiar de decisão abaixo de 0.5 elevou o recall da classe positiva de **~40%** para **~73.5%**,
> tornando o modelo muito mais adequado para uso clínico — onde falsos negativos (não detectar um óbito real)
> têm consequências graves.

A arquitetura mais simples `(16, 16)` superou configurações maiores, indicando que para 15 features categóricas
binárias, capacidade extra de representação não agrega valor significativo.

---

## Como Visualizar o Dashboard

```bash
# Abra o arquivo no navegador
start index.html        # Windows
open index.html         # macOS
xdg-open index.html     # Linux
```

O dashboard é um arquivo HTML estático — não requer servidor, build ou dependências.

---

## Documentação Completa

Consulte [`docs.html`](docs.html) para a documentação técnica completa em formato imprimível (PDF).

---

## Tecnologias

- **Python** — Scikit-learn, Pandas, NumPy
- **Dashboard** — HTML5 + CSS3 + JavaScript vanilla
- **Fontes** — Inter, Space Mono (Google Fonts)
- **Dataset** — Kaggle COVID-19 Dataset (México, 2020)
