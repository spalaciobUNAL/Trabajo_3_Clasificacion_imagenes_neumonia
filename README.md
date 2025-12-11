# Trabajo 3: Clasificacion Automatica de Neumonia en Radiografias de Torax

## Descripcion del Proyecto

Este proyecto implementa un sistema completo de clasificacion binaria de radiografias de torax (normales vs neumonia), comparando enfoques tradicionales de Machine Learning con descriptores handcrafted contra modelos de Deep Learning con Transfer Learning.

### Objetivos

1. **Analizar** la calidad y caracteristicas del dataset de radiografias de torax
2. **Extraer** descriptores tradicionales de imagenes (HOG, LBP, GLCM, Gabor, Momentos de Hu)
3. **Entrenar y comparar** clasificadores clasicos (SVM, Random Forest, k-NN, XGBoost)
4. **Implementar** Transfer Learning con ResNet para comparar con metodos tradicionales
5. **Evaluar** metricas de desempeno y robustez de todos los modelos

## Estructura del Proyecto

```
proyecto-clasificacion-neumonia/
├── README.md                   # Este archivo
├── Reporte.md                  # Reporte tecnico completo
├── requirements.txt            # Dependencias del proyecto
├── data/
│   ├── raw/                   # Dataset original de Kaggle
│   │   └── chest_xray/
│   │       ├── train/
│   │       │   ├── NORMAL/
│   │       │   └── PNEUMONIA/
│   │       ├── test/
│   │       └── val/
│   └── features/              # Caracteristicas extraidas (NPY, JSON)
├── src/
│   ├── preprocessing.py       # Pipeline de preprocesamiento (CLAHE, normalizacion)
│   ├── utils.py               # Funciones auxiliares y visualizacion
│   ├── download_dataset.py    # Script para descargar dataset de Kaggle
│   └── verify_dataset.py      # Verificacion de integridad del dataset
├── notebooks/
│   ├── 01_exploratory_analysis_clean.ipynb  # Analisis exploratorio
│   ├── 02_feature_extraction.ipynb          # Extraccion de descriptores
│   └── 03_classification.ipynb              # Clasificacion y comparacion
├── models/
│   └── model_configurations.json  # Configuraciones de modelos entrenados
└── results/
    ├── figures/               # Graficas y visualizaciones
    └── *.csv, *.png           # Resultados de clasificacion
```

## Instalacion

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- GPU con CUDA (opcional, para Deep Learning acelerado)

### Pasos de Instalacion

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/Trabajo_3_Clasificacion_imagenes_neumonia.git
cd Trabajo_3_Clasificacion_imagenes_neumonia
```

2. Crear un entorno virtual (recomendado):
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate
```

3. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

4. Descargar el dataset de Kaggle:
```bash
# Opcion 1: Usando el script incluido
python src/download_dataset.py

# Opcion 2: Manual desde Kaggle
# https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
# Descomprimir en data/raw/
```

## Uso

### 1. Analisis Exploratorio

```bash
jupyter notebook notebooks/01_exploratory_analysis_clean.ipynb
```

Este notebook permite:
- Visualizar muestras del dataset
- Analizar distribucion de clases (desbalance)
- Explorar tamanos y caracteristicas de las imagenes
- Aplicar y visualizar el pipeline de preprocesamiento
- Comparar distribuciones de intensidad entre clases (KDE, CDF)

### 2. Extraccion de Caracteristicas

```bash
jupyter notebook notebooks/02_feature_extraction.ipynb
```

Este notebook incluye:
- Extraccion de descriptores HOG (26,244 dimensiones)
- Calculo de Momentos de Hu (7 momentos)
- Extraccion de descriptores LBP (26 dimensiones)
- Calculo de caracteristicas GLCM (60 dimensiones)
- Aplicacion de filtros de Gabor (60 dimensiones)
- Estadisticas de primer orden (11 caracteristicas)
- Total: **26,415 dimensiones** de caracteristicas por imagen

### 3. Clasificacion y Comparacion

```bash
jupyter notebook notebooks/03_classification.ipynb
```

Este notebook implementa:
- Preprocesamiento y seleccion de caracteristicas
- Entrenamiento de clasificadores tradicionales (SVM, RF, k-NN, GB, LR, NB)
- Optimizacion de hiperparametros con Grid Search
- Evaluacion con matrices de confusion, ROC, AUC
- Implementacion de CNN y Transfer Learning (VGG16, ResNet)
- Comparacion final entre todos los enfoques

## Metodologia

### Dataset

- **Fuente:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Total de imagenes:** 5,856
  - Normales: 1,583
  - Neumonia: 4,273
- **Formato:** JPEG, resolucion variable (~1024x1024 px)
- **Division:** Train 70% / Test 30%

### Pipeline de Preprocesamiento

1. **Conversion a escala de grises**
2. **Redimensionamiento** a 224x224 pixeles (tamano estandar)
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
4. **Normalizacion** de intensidades al rango [0, 1]

### Descriptores Tradicionales

| Descriptor | Dimensiones | Descripcion |
|------------|-------------|-------------|
| HOG | 26,244 | Histogramas de gradientes orientados, captura bordes y estructuras |
| Momentos de Hu | 7 | Invariantes a rotacion, escala y traslacion |
| Contorno | 7 | Area, perimetro, excentricidad, solidez, extent |
| LBP | 26 | Patrones binarios locales, textura robusta a iluminacion |
| GLCM | 60 | Matriz de co-ocurrencia: contraste, homogeneidad, energia |
| Gabor | 60 | Filtros multi-frecuencia y orientacion |
| Estadisticas | 11 | Media, std, varianza, skewness, kurtosis, entropia |

### Clasificadores Tradicionales

- **SVM** (Support Vector Machine) con kernel RBF
- **Random Forest** (100-200 arboles)
- **k-NN** (k-Nearest Neighbors)
- **Gradient Boosting**
- **Logistic Regression**
- **Naive Bayes**

### Deep Learning

- **CNN Simple:** 3 capas convolucionales
- **CNN Profunda:** 4 capas convolucionales
- **VGG16 Transfer Learning:** Capas preentrenadas congeladas
- **ResNet Transfer Learning:** Skip connections para aprendizaje profundo

## Resultados

### Comparacion de Enfoques

| Modelo / Enfoque      | Accuracy | Precision | Recall  | F1-Score | AUC-ROC |
|-----------------------|---------|-----------|--------|----------|---------|
| ResNet Transfer (DL)  | 83.65%  | 79.88%    | 98.7%  | 88.3%    | 0.945   |
| k-NN (Tradicional)    | 79%     | 74.5%     | 88%    | 80.7%    | 0.873   |
| Random Forest         | 73%     | 71%       | 82%    | 76.5%    | 0.815   |
| XGBoost               | 68%     | 65%       | 78%    | 70%      | 0.78    |
| SVM                   | 64%     | 62%       | 76%    | 68%      | 0.755   |

### Hallazgos Principales

1. **ResNet Transfer Learning** logra el mejor rendimiento general con F1=88.3% y AUC=0.945
2. **k-NN** es el mejor clasificador tradicional con F1=80.7%
3. **HOG** es el descriptor individual mas discriminativo (100% de las top-20 caracteristicas)
4. Los descriptores de textura (LBP, GLCM, Gabor) combinados sin HOG tambien son efectivos
5. El desbalance de clases afecta mas a modelos tradicionales que a Deep Learning

## Uso Avanzado

### Preprocesamiento de Imagenes

```python
from src.preprocessing import preprocess_pipeline

# Aplicar pipeline completo
result = preprocess_pipeline(
    image,
    target_size=(224, 224),
    apply_clahe=True,
    normalize=True,
    segment_lungs=False  # Opcional: segmentacion pulmonar
)

processed_image = result['final']
```

### Extraccion de Caracteristicas

```python
# Importar funciones de extraccion (del notebook 02)
def extract_all_features(image):
    features = {}
    features['hog'] = extract_hog_features(image)
    features['hu_moments'] = extract_hu_moments(image)
    features['lbp'] = extract_lbp_features(image)
    features['glcm'] = extract_glcm_features(image)
    features['gabor'] = extract_gabor_features(image)
    features['statistics'] = extract_statistical_features(image)
    return features

# Extraer caracteristicas de una imagen
all_features = extract_all_features(processed_image)
```

### Parametros de Descriptores

```python
# HOG
hog_features = hog(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys'
)

# LBP
lbp = local_binary_pattern(image, n_points=24, radius=3, method='uniform')

# GLCM
glcm = graycomatrix(image, distances=[1, 2, 3], angles=[0, 45, 90, 135])
```

### Entrenamiento de Clasificadores

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Definir parametros para Grid Search
params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Entrenar con optimizacion
clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, params, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
```

## Solucion de Problemas

### Problema: Dataset no encontrado

**Solucion:**
- Verificar que el dataset este en `data/raw/chest_xray/`
- Ejecutar `python src/verify_dataset.py` para verificar estructura
- Descargar manualmente desde Kaggle si es necesario

### Problema: Memoria insuficiente para CNN

**Solucion:**
- Reducir `batch_size` en el entrenamiento
- Usar `max_samples_per_class` para limitar datos
- Activar `memory_growth` en GPU de TensorFlow

### Problema: F1-Score bajo en modelos tradicionales

**Solucion:**
- Aplicar tecnicas de balanceo (SMOTE, class_weight)
- Aumentar numero de caracteristicas seleccionadas (SelectKBest k > 100)
- Probar combinaciones de descriptores (textura vs forma)

### Problema: Overfitting en CNN

**Solucion:**
- Aumentar `dropout_rate` (0.5 o superior)
- Usar EarlyStopping con `patience=5`
- Aplicar data augmentation mas agresivo
- Reducir complejidad de la red

## Referencias

1. Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection*. CVPR.
2. Ojala, T., Pietikainen, M., & Maenpaa, T. (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns*. IEEE TPAMI.
3. Haralick, R. M. (1973). *Textural features for image classification*. IEEE TSMC.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR.
5. Pizer, S. et al. (1987). *Adaptive histogram equalization and its variations*. CVGIP.
6. Kermany, D. et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell.

## Estructura del Codigo

### Modulos Principales

- **`preprocessing.py`**: Pipeline de preprocesamiento con CLAHE, normalizacion, segmentacion
- **`utils.py`**: Funciones auxiliares para visualizacion, metricas, IO de datos
- **`download_dataset.py`**: Script para descarga automatica del dataset de Kaggle
- **`verify_dataset.py`**: Verificacion de integridad y estructura del dataset

### Notebooks

1. **`01_exploratory_analysis_clean.ipynb`**: Analisis exploratorio completo del dataset
2. **`02_feature_extraction.ipynb`**: Extraccion de todos los descriptores tradicionales
3. **`03_classification.ipynb`**: Clasificacion, optimizacion y comparacion de modelos

## Resultados Esperados

Despues de ejecutar el pipeline completo, obtendra:

1. **Figuras** en `results/figures/`:
   - `dataset_samples_train.png` - Muestras del dataset
   - `preprocessing_demo.png` - Pipeline de preprocesamiento
   - `hog_visualization.png` - Visualizacion de descriptores HOG
   - `lbp_features_demo.png` - Caracteristicas LBP
   - `gabor_features_demo.png` - Filtros de Gabor
   - `confusion_matrices.png` - Matrices de confusion
   - `roc_curves.png` - Curvas ROC
   - `final_comparison.png` - Comparacion de todos los modelos

2. **Resultados** en `results/`:
   - `classification_results_summary.csv` - Resumen de metricas
   - `descriptor_combinations_results.csv` - Resultados por combinacion
   - `final_comparison_all_approaches.csv` - Comparacion final

3. **Modelos** en `models/`:
   - Clasificadores entrenados (.pkl)
   - Configuraciones de hiperparametros (.json)

## Contribucion del Equipo

- **Henrry Uribe Cabrera Ordonez:** Reporte, Analisis de datos, extraccion de caracteristicas, entrenamiento y evaluacion de modelos.
- **Laura Sanin Colorado:** Preprocesamiento de imagenes, analisis exploratorio y documentacion de resultados.
- **Juan Manuel Sanchez Restrepo:** Implementacion de clasificadores tradicionales, ajuste de parametros y generacion de graficos de desempeno.
- **Sebastian Palacio Betancur:** Implementacion de Transfer Learning con ResNet, evaluacion de modelos de deep learning y comparacion con modelos tradicionales.

Este proyecto es parte del curso de Vision por Computador.

Profesor: Juan David Ospina Arango

Monitor: Andres Mauricio Zapata
