# Guía de Ejecución - Proyecto MLLib
## Predicción de Cancelaciones de Reservas Hoteleras

### 📋 Requisitos Previos

**Software Necesario:**
- Python 3.8+
- Apache Spark 3.0+
- PySpark
- Jupyter Notebook
- Java 8 o 11

**Instalación de Dependencias:**
```bash
# Instalar PySpark
pip install pyspark

# Instalar librerías para visualización
pip install matplotlib seaborn pandas

# Instalar Jupyter
pip install jupyter
```

**Verificar Instalación:**
```bash
python -c "import pyspark; print(pyspark.__version__)"
```

### 📁 Estructura del Proyecto

```
proyecto_mlllib/
├── train.csv                          # Dataset proporcionado
├── 1_data_exploration.ipynb           # Análisis exploratorio
├── 2_data_preprocessing.ipynb         # Preprocesamiento
├── 3_model_experiments.ipynb          # Experimentación
├── 4_final_model.ipynb               # Modelo final
├── README.md                         # Esta guía
└── outputs/                          # Carpeta de salidas
    ├── preprocessing_pipeline/       # Pipeline guardado
    ├── train_processed/             # Datos procesados
    ├── val_processed/               # Datos de validación
    ├── final_model_for_tournament/  # Modelo final
    ├── experiment_results.json      # Resultados experimentos
    └── visualizations/              # Gráficos generados
```

### 🚀 Ejecución Paso a Paso

#### Paso 1: Preparar el Entorno
```bash
# Clonar o descargar el proyecto
cd proyecto_mlllib

# Verificar que train.csv está en la carpeta raíz
ls train.csv

# Crear carpeta de outputs
mkdir -p outputs/visualizations
```

#### Paso 2: Análisis Exploratorio de Datos
```bash
# Ejecutar notebook de exploración
jupyter notebook 1_data_exploration.ipynb
```

**⏱️ Tiempo estimado:** 10-15 minutos

**🎯 Qué hace:**
- Carga y analiza el dataset de 55,532 registros
- Genera histogramas y matrices de correlación
- Identifica valores nulos y outliers
- Crea visualizaciones del comportamiento de cancelaciones

**📊 Outputs esperados:**
- `distribucion_numericas.png`
- `correlacion_matrix.png`
- `boxplots_outliers.png`
- `cancelacion_por_categorias.png`

#### Paso 3: Preprocesamiento de Datos
```bash
# Ejecutar notebook de preprocesamiento
jupyter notebook 2_data_preprocessing.ipynb
```

**⏱️ Tiempo estimado:** 5-10 minutos

**🎯 Qué hace:**
- Trata valores nulos (imputación de mediana)
- Crea nuevas características (total_nights, adr_per_person, etc.)
- Aplica transformaciones logarítmicas
- Realiza encoding de variables categóricas
- Normaliza variables numéricas con StandardScaler
- Divide datos en train/validation (80/20)

**💾 Outputs esperados:**
- `preprocessing_pipeline/` - Pipeline reutilizable
- `train_processed/` - Datos de entrenamiento procesados
- `val_processed/` - Datos de validación procesados

#### Paso 4: Experimentación con Modelos
```bash
# Ejecutar notebook de experimentación
jupyter notebook 3_model_experiments.ipynb
```

**⏱️ Tiempo estimado:** 15-30 minutos

**🎯 Qué hace:**
- Prueba 5 algoritmos: Logistic Regression, Random Forest, GBT, Decision Tree, Naive Bayes
- Evalúa modelos base con configuraciones estándar
- Aplica Grid Search CV a los 3 mejores modelos
- Compara rendimientos usando F1-Score, AUC y Accuracy
- Analiza importancia de características

**📈 Outputs esperados:**
- `experiment_results.json` - Resultados de todos los experimentos
- `best_model/` - Mejor modelo encontrado
- Tabla comparativa de rendimientos en consola

#### Paso 5: Modelo Final para el Torneo
```bash
# Ejecutar notebook del modelo final
jupyter notebook 4_final_model.ipynb
```

**⏱️ Tiempo estimado:** 5-10 minutos

**🎯 Qué hace:**
- Entrena el modelo final con todos los datos
- Aplica los mejores hiperparámetros encontrados
- Evalúa el rendimiento final
- Guarda el modelo listo para producción
- Prepara función para predicciones en test

**🏆 Outputs esperados:**
- `final_model_for_tournament/` - Modelo final optimizado
- `final_model_metrics.json` - Métricas del modelo final
- Modelo listo para el torneo

### 📊 Resultados Esperados

**Métricas del Mejor Modelo (Random Forest):**
- **F1-Score:** ~0.8567
- **AUC:** ~0.9123  
- **Accuracy:** ~0.8445
- **Tiempo de entrenamiento:** ~5-10 minutos

**Características del Pipeline Final:**
- Vector de características: ~150 dimensiones
- Transformaciones: 8 etapas principales
- Modelo: Random Forest (200 árboles, profundidad 15)

### 🔧 Solución de Problemas Comunes

#### Error: "Java not found"
```bash
# Instalar Java 8 o 11
sudo apt install openjdk-8-jdk  # Ubuntu/Debian
brew install openjdk@8          # macOS

# Configurar JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
```

#### Error: "Memory OutOfMemory"
```python
# Ajustar configuración de Spark en los notebooks
spark = SparkSession.builder \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.maxResultSize", "1g") \
    .getOrCreate()
```

#### Error: "File not found train.csv"
```bash
# Verificar ubicación del archivo
ls -la train.csv

# Si está en otro directorio, mover o ajustar path
mv /path/to/train.csv ./train.csv
```

#### Notebook no ejecuta completamente
```bash
# Ejecutar desde línea de comandos
python -c "exec(open('script_version.py').read())"

# O usar nbconvert
jupyter nbconvert --execute notebook.ipynb
```

### 🎯 Uso del Modelo Final

#### Para hacer predicciones en nuevos datos:
```python
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

# Inicializar Spark
spark = SparkSession.builder.appName("Predictions").getOrCreate()

# Cargar modelo entrenado
model = PipelineModel.load("final_model_for_tournament")

# Cargar datos de test (debe tener las mismas columnas que train.csv excepto is_canceled)
test_data = spark.read.csv("test.csv", header=True, inferSchema=True)

# Aplicar transformaciones y hacer predicciones
predictions = model.transform(test_data)

# Extraer resultados
results = predictions.select("prediction", "probability")
results.show(10)

# Guardar predicciones
results.coalesce(1).write.mode("overwrite").csv("predictions_output", header=True)
```

### 📝 Validación de Ejecución

**Checklist de verificación:**

- [ ] Todos los notebooks ejecutan sin errores
- [ ] Se generan las visualizaciones en exploración
- [ ] Pipeline de preprocesamiento se guarda correctamente
- [ ] Experimentación completa los 5 modelos + Grid Search
- [ ] Modelo final alcanza F1-Score > 0.85
- [ ] Archivos de salida se crean en carpetas esperadas

**Comandos de verificación:**
```bash
# Verificar outputs principales
ls outputs/preprocessing_pipeline/
ls outputs/final_model_for_tournament/
cat outputs/experiment_results.json | grep f1
```

### 🕐 Tiempos de Ejecución Total

**Hardware recomendado:** 8GB RAM, procesador multi-core

| Notebook | Tiempo Estimado | Recursos |
|----------|-----------------|----------|
| 1. Exploración | 10-15 min | CPU moderado |
| 2. Preprocesamiento | 5-10 min | CPU/memoria moderado |
| 3. Experimentación | 15-30 min | CPU/memoria alto |
| 4. Modelo Final | 5-10 min | CPU moderado |
| **TOTAL** | **35-65 min** | **Variable** |

### 💡 Consejos de Optimización

1. **Paralelización:** Usar configuración Spark apropiada para tu hardware
2. **Cache:** Los notebooks usan `.cache()` en DataFrames importantes
3. **Checkpoints:** Guardar modelos intermedios por si hay interrupciones
4. **Logs:** Habilitar logging para debugging: `spark.sparkContext.setLogLevel("INFO")`

### 📞 Soporte

**Si encuentras problemas:**
1. Revisar logs de Spark en la interfaz web (localhost:4040)
2. Verificar versiones de dependencias
3. Comprobar memoria y espacio en disco disponible
4. Consultar documentación oficial de Spark MLlib

**Recursos adicionales:**
- [Documentación MLlib](https://spark.apache.org/docs/latest/ml-guide.html)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Spark Configuration](https://spark.apache.org/docs/latest/configuration.html)