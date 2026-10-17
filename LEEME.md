# Predicción de Fuga de Clientes en Telecomunicaciones

> English version: [README.md](README.md)

Un proyecto de clasificación binaria que predice **qué clientes de un operador de telecomunicaciones probablemente cancelarán su servicio (churn)**, para que la empresa pueda ofrecerles promociones antes de que se vayan.

El proyecto integra cuatro fuentes de datos, construye una variable de **antigüedad del cliente** sin fuga de información y compara **12 configuraciones de modelos** (4 modelos × 3 estrategias para el desbalance de clases) antes de ajustar la mejor. El modelo seleccionado, **LightGBM con oversampling de la clase minoritaria**, obtuvo un **ROC-AUC medio de 0.9303 en validación cruzada de 5 folds** y **0.9411 de ROC-AUC con 0.79 de Recall en el conjunto de prueba independiente**, muy por encima del requisito de **ROC-AUC ≥ 0.75**.

---

# Contexto de Negocio

Interconnect, un operador de telecomunicaciones, quiere reducir la fuga de clientes. Si la empresa puede identificar con anticipación a los clientes que están por irse, el equipo de marketing puede ofrecerles promociones o planes especiales. Un modelo confiable ayuda a:

* Enfocar las campañas de retención en los clientes con mayor riesgo.
* Reducir el costo de ofrecer descuentos a clientes que se quedarían de todos modos.
* Entender qué servicios y condiciones de contrato se asocian con la cancelación.

El modelo debe predecir un objetivo:

| Objetivo | Descripción |
| --- | --- |
| `churn` | 1 si el cliente canceló el servicio (`EndDate` es una fecha), 0 si sigue activo (`EndDate = "No"`) |

---

# Conceptos de Modelado de Churn

| Término | Significado |
| --- | --- |
| Churn | Un cliente que cancela el servicio |
| Desbalance de clases | Solo cerca del 26.5% de los clientes canceló, así que un modelo puede parecer preciso prediciendo "no churn" para todos |
| Fuga de información | Usar información que no estaría disponible al momento de predecir (por ejemplo, la fecha de cancelación) |
| Antigüedad (tenure) | Cuánto tiempo lleva el cliente con la empresa |
| `class_weight` / `scale_pos_weight` | Dar más peso a la clase minoritaria durante el entrenamiento |
| Oversampling | Repetir filas de la clase minoritaria en el conjunto de entrenamiento para equilibrar ambas clases |
| División estratificada | Mantener la misma proporción de churn en entrenamiento, validación y prueba |

## Grupos de Variables

```text
Contrato:    type, paperless_billing, payment_method, monthly_charges, total_charges
Personal:    gender, senior_citizen, partner, dependents
Internet:    internet_service, online_security, online_backup, device_protection,
             tech_support, streaming_tv, streaming_movies
Teléfono:    multiple_lines
Construida:  tenure_months
```

---

# Métrica de Evaluación

La métrica principal es el **ROC-AUC** (área bajo la curva ROC). Mide qué tan bien el modelo **ordena** a los clientes que cancelan por encima de los que no, considerando todos los umbrales de clasificación posibles, así que no depende del umbral elegido y es robusta al desbalance de clases.

El **Recall** se usa como métrica secundaria porque, para la retención, no detectar a un cliente que se va cuesta más que contactar a uno que se habría quedado:

$$
\text{Recall} = \frac{\text{Verdaderos Positivos}}{\text{Verdaderos Positivos} + \text{Falsos Negativos}}
$$

El requisito del proyecto era **ROC-AUC ≥ 0.75** en el conjunto de prueba. También se registraron F1, Precision y Accuracy para interpretar cada modelo.

---

# Dataset

Los datos provienen de cuatro archivos relacionados por `customerID`.

| Archivo | Filas | Columnas | Descripción |
| --- | ---: | ---: | --- |
| `datasets/final_provider/contract.csv` | 7,043 | 8 | Tipo de contrato, facturación, método de pago, cargos, fechas de inicio y fin |
| `datasets/final_provider/personal.csv` | 7,043 | 5 | Datos demográficos del cliente |
| `datasets/final_provider/internet.csv` | 5,517 | 8 | Servicios de internet (DSL o fibra óptica, seguridad, respaldo, streaming…) |
| `datasets/final_provider/phone.csv` | 6,361 | 2 | Servicio telefónico (líneas múltiples) |

* **7,043 clientes** en total: 1,526 sin servicio de internet y 682 sin servicio telefónico.
* Los clientes iniciaron entre **octubre de 2013 y febrero de 2020**; todas las cancelaciones ocurrieron entre **octubre de 2019 y enero de 2020**.
* Tasa de churn: **1,869 clientes (26.5%)** cancelaron y **5,174 (73.5%)** se quedaron.

**Origen de los datos.** Los datasets fueron proporcionados por **TripleTen** como parte de su Data Science Bootcamp y se incluyen en este repositorio con fines educativos y de reproducibilidad.

---

# Flujo del Proyecto

```text
Datos originales (4 archivos)
            │
            ▼
1. Preparación de datos
   ├── Nombres en snake_case, fechas, total_charges numérico
   ├── Objetivo: churn = EndDate distinto de "No"
   └── Uniones left por customer_id + valores faltantes de las uniones
            │
            ▼
2. Ingeniería de variables: tenure_months (sin información futura)
            │
            ▼
3. División estratificada: 75% entrenamiento / 15% validación / 10% prueba + One-Hot Encoding
            │
            ▼
4. Modelado
   ├── Experimento 1: 4 modelos × 3 estrategias de desbalance (12 configuraciones)
   ├── Experimento 2: LightGBM con y sin tenure_months
   ├── Experimento 3: búsquedas de hiperparámetros de LightGBM
   ├── Validación cruzada estratificada de 5 folds (oversampling dentro de cada fold)
   └── Reentrenar con entrenamiento + validación y evaluar una vez en prueba
```

---

# 1. Preparación de Datos

* Los nombres de columnas se convirtieron a `snake_case`.
* `begin_date` y `end_date` se convirtieron a fecha (formato `AAAA-MM-DD`). `end_date = "No"` significa que el cliente sigue activo.
* `total_charges` tenía **11 valores vacíos**, todos de clientes que se dieron de alta el 1 de febrero de 2020 y aún no recibían su primera factura. Se asignaron a **0**.
* Las columnas "Yes"/"No" se convirtieron a 1/0.
* Las cuatro tablas se unieron a `contract` con uniones left por `customer_id`, verificando relaciones uno a uno (`validate='one_to_one'`).
* Los valores faltantes generados por las uniones significan que el cliente no tiene ese servicio:
  * Columnas de servicios de internet → **0**, e `internet_service` → **"No internet"** (1,526 clientes).
  * `multiple_lines` → **0** (682 clientes sin servicio telefónico).
* `customer_id`, `begin_date`, `end_date` y `churn` se excluyeron de las variables.

---

# 2. Ingeniería de Variables: Antigüedad del Cliente

`tenure_months` estima cuánto tiempo lleva cada cliente con la empresa. Se calculó **sin usar la fecha de cancelación**:

| Opción | Qué representa | ¿Se puede usar? |
| --- | --- | --- |
| `end_date − begin_date` | Duración final de la relación, conocida solo después de que el cliente se va | ❌ Información futura |
| `end_date` | Cuándo ocurrió la cancelación | ❌ Fuga directa |
| `fecha_corte − begin_date` | Antigüedad hasta una fecha de corte común | ✅ |

La fecha de corte es la `begin_date` más reciente de los datos (**2020-02-01**):

```python
fecha_corte = df['begin_date'].max()
df['tenure_months'] = (fecha_corte - df['begin_date']).dt.days / 30.44
```

La variable resultante va de **0 a 76 meses**, con una media de **33.1 meses**.

---

# 3. División y Codificación

Los datos se dividieron con `stratify` para que cada conjunto conserve la tasa de churn de 26.5%:

```text
100% de los datos (7,043 clientes)
│
├── 75% ENTRENAMIENTO ──→ 5,282 clientes → entrenamiento de modelos
├── 15% VALIDACIÓN ─────→ 1,056 clientes → selección de modelo e hiperparámetros
└── 10% PRUEBA ─────────→   705 clientes → evaluación final (se usa una sola vez)
```

Las variables categóricas (`gender`, `type`, `payment_method`, `internet_service`) se codificaron con `OneHotEncoder(handle_unknown='ignore')` dentro de un `ColumnTransformer`, ajustado **solo con el conjunto de entrenamiento**. La matriz final tiene **26 variables**.

Para el oversampling, las filas de churn del conjunto de entrenamiento se repitieron 3 veces (de 1,402 a 4,206 clientes que cancelan contra 3,880 que no). Validación y prueba nunca se modificaron.

---

# 4. Modelado

## Experimento 1: Modelos y Estrategias de Desbalance

Se entrenaron cuatro modelos con tres estrategias cada uno: **sin balanceo**, **pesos de clase** (`class_weight='balanced'`, o `scale_pos_weight ≈ 2.77` en LightGBM) y **oversampling**.

| Modelo | Sin balanceo | Pesos de clase | Oversampling |
| --- | ---: | ---: | ---: |
| **LightGBM** | 0.9239 | 0.9201 | **0.9274** |
| Random Forest | 0.8607 | 0.8581 | 0.8607 |
| Logistic Regression | 0.8344 | 0.8339 | 0.8337 |
| LinearSVC | 0.8320 | 0.8333 | 0.8328 |

*ROC-AUC en validación.*

* **LightGBM** fue claramente el mejor modelo, lo que sugiere relaciones no lineales entre las variables y el churn.
* El balanceo **aumentó el Recall** en todos los modelos (en Logistic Regression, de cerca de 0.54 a 0.77) a costa de menor Precision, mientras que **el ROC-AUC casi no cambió**: el balanceo mueve el umbral de decisión, pero el orden de los clientes se mantiene casi igual.
* No se usó Accuracy como métrica principal: los modelos sin balanceo tuvieron la Accuracy más alta porque predecían casi siempre la clase mayoritaria.

## Experimento 2: Impacto de `tenure_months`

| LightGBM (oversampling) | ROC-AUC en Validación |
| --- | ---: |
| Con `tenure_months` | **0.9274** |
| Sin `tenure_months` | 0.8256 |

Quitar la antigüedad redujo el ROC-AUC en cerca de **0.10**, así que la variable se conservó.

## Experimento 3: Ajuste de Hiperparámetros (LightGBM + oversampling)

| Búsqueda | Qué se varió | Mejor configuración | Mejor ROC-AUC en Validación |
| --- | --- | --- | ---: |
| 1 | `n_estimators` (100–300) y `num_leaves` (15, 31, 63), `learning_rate = 0.05` | 300 árboles, 31 hojas | 0.9314 |
| 2 | `min_child_samples` de 10 a 100, con 300 árboles y 63 hojas | `min_child_samples = 75` | 0.9355 |
| 3 | `min_child_samples` de 40 a 90 alrededor del mejor valor | `min_child_samples = 75` | 0.9355 |

En la búsqueda 1, 31 hojas dieron un ROC-AUC un poco mayor que 63 hojas (0.9314 vs 0.9296). En la segunda búsqueda se mantuvieron **63 hojas** porque los árboles más grandes se pueden regularizar con `min_child_samples`; con `min_child_samples = 75` superaron a la primera búsqueda (0.9355).

Configuración final:

```text
LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
               max_depth=-1, min_child_samples=75, random_state=42)
```

## Validación Cruzada

Antes de usar el conjunto de prueba, la configuración final se revisó con **validación cruzada estratificada de 5 folds** sobre el conjunto de entrenamiento. El oversampling se aplicó **dentro de cada fold**, solo en su parte de entrenamiento, para que ninguna fila duplicada aparezca a la vez en entrenamiento y validación.

| Fold | ROC-AUC |
| ---: | ---: |
| 1 | 0.9222 |
| 2 | 0.9231 |
| 3 | 0.9254 |
| 4 | 0.9515 |
| 5 | 0.9295 |
| **Media ± desv. est.** | **0.9303 ± 0.0121** |

## Resultado Final

El modelo se reentrenó con **entrenamiento + validación** (con oversampling) y se evaluó **una sola vez** en el conjunto de prueba:

```text
Modelo:              LightGBM + oversampling
ROC-AUC en CV:       0.9303 ± 0.0121
ROC-AUC en prueba:   0.9411
Recall en prueba:    0.7914
F1 en prueba:        0.8000
Precision en prueba: 0.8087
Accuracy en prueba:  0.8950
Requisito:           ROC-AUC ≥ 0.75   ✓
```

En los 705 clientes de prueba, el modelo identificó cerca del **79% de los 187 clientes que cancelaron**, y cerca del 81% de los clientes que marcó como churn realmente se fueron.

---

# Conclusiones

1. **Integrar las cuatro fuentes requirió interpretar bien los valores faltantes**: un cliente que no aparece en la tabla de internet o de teléfono simplemente no tiene ese servicio.

2. **`tenure_months` fue la variable más informativa.** Construirla con una fecha de corte común, en lugar de la fecha de cancelación, evitó la fuga de información; quitarla bajó el ROC-AUC de 0.927 a 0.826.

3. **LightGBM superó claramente a los demás modelos** (ROC-AUC en validación de alrededor de 0.92–0.93, contra 0.86 de Random Forest y 0.83 de los modelos lineales).

4. **El balanceo de clases cambió sobre todo el Recall, no el ROC-AUC.** El oversampling dio el mejor ROC-AUC para LightGBM y se conservó en el modelo final.

5. **El resultado es estable y supera el requisito**: la validación cruzada dio 0.9303 ± 0.0121 y el conjunto de prueba dio **0.9411 de ROC-AUC con 0.79 de Recall**.

## Interpretación de Negocio

Con un Recall de cerca de 79%, el modelo detecta aproximadamente 79 de cada 100 clientes que van a irse, así que el equipo de marketing puede priorizarlos para ofertas, cambios de plan o una llamada directa. Con el umbral actual, cerca del 21% de los clientes que cancelan no se detectaría, por lo que el modelo debe apoyar las decisiones de retención, no reemplazarlas.

## Siguientes Pasos

* Ajustar el umbral de clasificación según el costo de una oferta de retención frente al costo de perder a un cliente.
* Analizar la importancia de las variables para explicar qué condiciones de contrato y servicios impulsan el churn.
* Nota: una versión anterior del notebook leía las fechas como `AAAA-DD-MM`. La versión actual usa el formato correcto; los resultados de LightGBM no cambiaron, porque el error conservaba el orden de `tenure_months`.

---

# Estructura del Proyecto

```text
telecom_churn_prediction/
│
├── datasets/
│   └── final_provider/
│       ├── contract.csv
│       ├── internet.csv
│       ├── personal.csv
│       └── phone.csv
│
├── notebooks/
│   └── telecom_churn_prediction.ipynb
│
├── src/
│   └── funciones_personales.py
│
├── .gitignore
├── environment.yml
├── requirements.txt
├── LEEME.md
└── README.md
```

---

# Instalación

Clona el repositorio:

```bash
git clone https://github.com/pablo-vc87/telecom_churn_prediction.git

cd telecom_churn_prediction
```

## Opción 1: Conda

```bash
conda env create -f environment.yml
conda activate telecom_churn_prediction
jupyter notebook
```

## Opción 2: pip

Crea y activa un entorno virtual, y después:

```bash
pip install -r requirements.txt
jupyter notebook
```

Abre `notebooks/telecom_churn_prediction.ipynb` y ejecuta todas las celdas.

---

# Tecnologías

* Python
* Pandas
* NumPy
* scikit-learn
* LightGBM
* Jupyter Notebook

---

# Habilidades Demostradas

* Integración de múltiples fuentes de datos
* Interpretación de valores faltantes después de uniones
* Ingeniería de variables sin fuga de información (antigüedad del cliente)
* Clasificación binaria con desbalance de clases
* Pesos de clase y oversampling
* One-Hot Encoding con `ColumnTransformer`
* Comparación de modelos: Logistic Regression, LinearSVC, Random Forest y LightGBM
* Ajuste de hiperparámetros
* Validación cruzada estratificada con oversampling dentro de los folds
* Evaluación con ROC-AUC, Recall, Precision y F1

---

## Autor

**Pablo Andrés Vázquez Calva**

Data Scientist

GitHub: https://github.com/pablo-vc87

LinkedIn: https://www.linkedin.com/in/pablo-avc/
