# Sistema de deteccion de fraude en transacciones con 98% de AUC ROC
<img width="731" height="346" alt="image" src="https://github.com/user-attachments/assets/f67c5f12-8372-4fbe-87e9-d2549e6320ed" />



## Instalación

1. Clona el repositorio y entra en la carpeta del proyecto:

--> Ejecuta en la terminal:
git clone https://github.com/hugoperezgarcia/fraud-detection.git
cd fraud-detection

2. Descomprime el zip que hay en data/

3. Crea y activa un entorno virtual con las librerias

--> Ejecuta
python3 -m venv venv
source venv/bin/activate ##(En linux)
pip install -r requirements.txt

4. Ejecutar la aplicacion
--> Ejecuta
python3 -m scripts.train

5. Ver metricas
Al terminar en la consola saldran las metricas y en /models se habran guardado los modelos que se han creado.


## Descripcion
Este proyecto aborda el problema clásico de **detección de fraude en transacciones de tarjetas de crédito** mediante técnicas de Machine Learning.

El dataset contiene más de **284.000 transacciones**, de las cuales solo un **0.17%** son fraudulentas.  
Cada transacción está representada por **28 variables anónimas (V1–V28)** obtenidas mediante un proceso de *PCA* por motivos de confidencialidad, junto con las columnas `Time`, `Amount` y la variable objetivo `Class` (0 = no fraude, 1 = fraude).

El EDA se realizó en `notebooks/01_eda.ipynb` e incluyó:

- **Distribución de clases:**  
  Se confirmó un desbalance extremo (492 fraudes frente a 284.315 no fraudes).  
  Se analizó mediante gráficos de barras y conteo de clases.

- **Distribución por variable:**  
  Se graficaron las densidades (`KDE plots`) de cada variable (`V1`–`V28`) separadas por clase, para evaluar su poder discriminativo.  
  Algunas componentes como `V14`, `V17` y `V4` mostraron mayor capacidad de separación entre clases.

- **Correlaciones:**  
  Se generó una matriz de correlaciones para identificar relaciones entre variables y con la clase objetivo.  
  Las variables principales no mostraron multicolinealidad significativa.

- **Escalado de variables:**  
  Se detectó que las columnas `Amount` y `Time` necesitaban normalización para que el modelo pudiera tratarlas correctamente.

A partir de este análisis se definió el preprocesamiento estándar y las columnas a escalar.

## Pipeline

El flujo completo se ejecuta con un solo script (`scripts/train.py`) y consta de:

1. **Carga de datos**  
   Lectura desde `data/raw/creditcard.csv`.

2. **División del dataset**  
   Separa en *train*, *validation* y *test* con estratificación por clase.

3. **Preprocesamiento**  
   Estandarización de `Time` y `Amount` mediante `StandardScaler`.

4. **Entrenamiento con XGBoost**  
   Modelo binario (`binary:logistic`) con hiperparámetros básicos y ajuste de desbalance usando  
   `scale_pos_weight = (negativos / positivos)`.

5. **Evaluación del modelo**  
   Se calculan métricas de rendimiento (AUC, matriz de confusión, *classification report*).  
   El modelo típico alcanza un **AUC > 0.99** en test, indicando excelente capacidad de discriminación.

6. **Guardado de artefactos**  
   Se almacenan en `models/`:
   - `xgb_model.pkl`
   - `preprocess.pkl`

## Autor
Hugo Pérez
