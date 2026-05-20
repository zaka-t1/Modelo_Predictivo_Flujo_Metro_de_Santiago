[README_metro.md](https://github.com/user-attachments/files/28064446/README_metro.md)
# 🚇 Modelo Predictivo de Afluencia — Metro de Santiago

Sistema de predicción de flujo de pasajeros por estación y tramo horario, desarrollado con datos reales del **DTPM (Directorio de Transporte Público Metropolitano)** de Santiago de Chile.

---

## 📌 Descripción

Este proyecto desarrolla un modelo predictivo capaz de estimar la cantidad de pasajeros que utilizan cada estación del Metro de Santiago en distintos tramos horarios. Se aplicaron técnicas de **feature engineering**, comparación de múltiples algoritmos y evaluación con métricas estándar de regresión.

El modelo final alcanzó un **R² = 0.87** con un error promedio de **28 pasajeros por predicción**.

---

## 🎯 Resultados

| Modelo | R² | MAE |
|---|---|---|
| Regresión Lineal | - | - |
| Regresión Logística | - | - |
| Red Neuronal (MLP) | - | - |
| **Random Forest** ✅ | **0.87** | **~28 pasajeros** |

> **Modelo seleccionado:** Random Forest — mejor balance entre precisión y generalización.

---

## 🔧 Tecnologías utilizadas

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## 📁 Estructura del proyecto

```
├── modelo_predictivo_metro.py   # Script principal del modelo
├── metro_santiago_dataset_limpio.csv  # Dataset procesado (fuente: DTPM)
├── Mapa de Calor Metro.jpeg     # Visualización de correlaciones
├── Figure_1.png                 # Distribución de pasajeros por estación
├── Figure_2.png                 # Comparación de algoritmos
├── Figure_3.png                 # Curva de aprendizaje
├── Figure_4.png                 # Feature importance
├── Figure_7.png                 # Predicciones vs valores reales
├── Figure_9.png                 # Residuos del modelo
├── Figure_10.png                # Métricas finales
└── README.md
```

---

## ⚙️ Feature Engineering

Las variables más relevantes creadas para el modelo:

- `es_hora_punta` — clasificación binaria de tramos horarios críticos
- `tipo_estacion` — combinación / terminal / intermedia
- `Label Encoding` — codificación de variables categóricas (línea, estación)

---

## 📊 Visualizaciones

### Mapa de Calor por Estación
![Mapa de Calor](Mapa%20de%20Calor%20Metro.jpeg)

---

## 🚀 Cómo ejecutar

```bash
# Clonar el repositorio
git clone https://github.com/zaka-t1/Modelo_Predictivo_Flujo_Metro_de_Santiago.git
cd Modelo_Predictivo_Flujo_Metro_de_Santiago

# Instalar dependencias
pip install pandas numpy scikit-learn matplotlib

# Ejecutar el modelo
python modelo_predictivo_metro.py
```

---

## 👥 Equipo

Proyecto desarrollado en equipo de 3 integrantes — Ramo **Data Science**, Ingeniería Informática, UTEM (2025).

---

## 📄 Fuente de datos

Dataset obtenido del **DTPM — Directorio de Transporte Público Metropolitano** de Santiago de Chile.

---

*Desarrollado por [Zacarías Mora Torres](https://github.com/zaka-t1) · Ingeniería Informática UTEM*
