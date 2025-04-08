# 🌸 Classificador de Espécies da Flor Íris com FastAPI e Random Forest

Este projeto utiliza o framework **FastAPI** para disponibilizar uma API que prevê a espécie de uma flor íris com base em suas características morfológicas. O modelo de machine learning foi treinado com **Random Forest** usando o famoso dataset **Iris**, feito enquanto para a minha Sprint 3 enquanto estudava na FIAP.

---

## 📁 Estrutura do Projeto

O projeto é dividido em duas partes:

1. **Treinamento do Modelo (`model_traning.py`)**
2. **Serviço Web com FastAPI (`main.py`)**

---

## 🧠 Treinamento do Modelo

### 🔍 Etapas do Treinamento

- **Leitura dos dados** a partir do arquivo `Iris.csv`
- **Separação das features** (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) e da **classe alvo** (`Species`)
- **Divisão** dos dados em treino e teste
- **Treinamento** com o algoritmo `RandomForestClassifier`
- **Exportação do modelo** treinado usando `joblib`

```python
{`
model = RandomForestClassifier()
data = pd.read_csv('Iris.csv')

X_data = data.drop('Species', axis=1)
y_data = data['Species']
X_ach_traning, X_testing, y_ach_traning, y_testing = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

model.fit(X_ach_traning, y_ach_traning)
dump(model, 'archive_model_traning.joblib')
`}
```

---

## 🚀 API com FastAPI

### 📦 Endpoints

- `POST /arch_preview`  
  Recebe os dados da flor (comprimento e largura das sépalas e pétalas) e retorna a **espécie prevista** pelo modelo.

### 🔄 Estrutura esperada no corpo da requisição:

```json
{`
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
`}
```bash

### 📤 Resposta esperada:

```json
{`
{
  "Preview do POST": "Iris-setosa"
}
`}
```

### 🧩 Código do Servidor FastAPI

```python
{`
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

model = load('archive_model_traning.joblib')
app = FastAPI()

class ArchData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/arch_preview")
def arch_preview(data: ArchData):
    arch_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    arch_preview = model.predict(arch_data)
    return {"Preview do POST": arch_preview[0]}
`}
```

> **Atenção:** no código original havia um erro no uso da função `preview()`, o correto é `predict()`.

---

## 🧪 Como Executar

### ⚙️ Instalação de Dependências

```bash
pip install fastapi uvicorn scikit-learn pandas joblib
```

## 🚦 Executar o servidor FastAPI
```bash
uvicorn main:app --reload
```
