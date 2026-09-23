import re

def to_snake_case(col):
    # Buscapalabra normal seguida de mayúsculas (customerID e inserta '_' entre ambos
    col = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', col)
    # munúsculas a las que les sigan mayúscvulas e inserta '_' entre ambos (TotalCharges)
    col = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', col)
    # cambia todo a minúsculas
    return col.lower()



#======================================
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    roc_auc_score, 
    classification_report, 
    accuracy_score,
    recall_score,
    precision_score
)


def evaluar_modelo_clasificacion(
    modelo,
    X_train,
    y_train,
    X_valid,
    y_valid,
    nombre_modelo,
    estrategia_balance,
    parametros=None
):
    """
        Entrena un modelo de clasificación y devuelve sus métricas de desempeño.
    
        Parámetros
        ----------
        modelo : estimador de sklearn o compatible
        X_train, y_train : datos de entrenamiento
        X_valid, y_valid : datos de validación
        nombre_modelo : str
        parametros : dict, opcional
            Diccionario con los hiperparámetros utilizados.
    
        Retorna
        -------
        DataFrame con una fila de resultados.
        """

    # ==========================================
    # Entrenamiento
    # ==========================================

    inicio_train = time.perf_counter()
    
    modelo.fit(X_train, y_train)
    
    tiempo_train = time.perf_counter() - inicio_train


    # ==========================================
    # Predicción
    # ==========================================

    inicio_pred = time.perf_counter()

    predicciones = modelo.predict(X_valid)

    tiempo_pred = (time.perf_counter() - inicio_pred)


    # ==========================================
    # Métricas de clasificación
    # ==========================================

    f1 = f1_score(
        y_valid,
        predicciones,
        zero_division=0
    )

    precision = precision_score(
        y_valid,
        predicciones,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        predicciones,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_valid,
        predicciones
    )


    # ==========================================
    # ROC-AUC
    # ==========================================

    roc_auc = np.nan

    if hasattr(modelo, 'predict_proba'):

        scores = modelo.predict_proba(X_valid)[:, 1]

        roc_auc = roc_auc_score(
            y_valid,
            scores
        )

    elif hasattr(modelo, 'decision_function'):

        scores = modelo.decision_function(X_valid)

        roc_auc = roc_auc_score(
            y_valid,
            scores
        )


    # ==========================================
    # Formato de hiperparámetros
    # ==========================================

    if parametros is None or len(parametros) == 0:
        parametros_txt = 'Default'
    else:
        parametros_txt = ', '.join(
            f'{k}={v}'
            for k, v in parametros.items()
        )


    # ==========================================
    # Dataframe de resultados
    # ==========================================

    resultado = pd.DataFrame({
        'Modelo': [nombre_modelo],
        'Balance': [estrategia_balance],
        'Parámetros': [parametros_txt],
        'F1': [f1],
        'Precision': [precision],
        'Recall': [recall],
        'Accuracy': [accuracy],
        'ROC-AUC': [roc_auc],
        'Tiempo entrenamiento (s)': [tiempo_train],
        'Tiempo predicción (s)': [tiempo_pred]
    })


    return resultado, modelo
    
#====================================================
def probar_hiperparametros_clasificacion(
    modelo_base,
    lista_parametros,
    X_train,
    y_train,
    X_valid,
    y_valid,
    nombre_modelo
):
    """
    Prueba diferentes combinaciones de
    hiperparámetros y devuelve los resultados
    ordenados por F1.
    """

    resultados = []

    mejor_modelo = None

    mejor_f1 = -1


    for parametros in lista_parametros:

        # ==========================================
        # Crear modelo
        # ==========================================

        modelo = modelo_base(
            **parametros
        )


        # ==========================================
        # Evaluar modelo
        # ==========================================

        resultado, modelo_entrenado = (
            evaluar_modelo_clasificacion(

                modelo=modelo,

                X_train=X_train,
                y_train=y_train,

                X_valid=X_valid,
                y_valid=y_valid,

                nombre_modelo=nombre_modelo,

                parametros=parametros
            )
        )


        resultados.append(
            resultado
        )


        # ==========================================
        # Guardar mejor modelo
        # ==========================================

        f1_actual = resultado.loc[
            0,
            'F1'
        ]


        if f1_actual > mejor_f1:

            mejor_f1 = f1_actual

            mejor_modelo = (
                modelo_entrenado
            )


    # ==========================================
    # Unir resultados
    # ==========================================

    resultados = (

        pd.concat(
            resultados,
            ignore_index=True
        )

        .sort_values(
            'F1',
            ascending=False
        )

        .reset_index(
            drop=True
        )

    )


    return resultados, mejor_modelo


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report, roc_curve, precision_recall_curve, precision_recall_fscore_support
#=============================================================
def probar_hiperparametros_por_auc(
    modelo_base,
    lista_parametros,
    X_train,
    y_train,
    X_valid,
    y_valid,
    nombre_modelo,
    estrategia_balance
):
    resultados = []

    mejor_modelo = None
    mejor_auc = -np.inf

    for parametros in lista_parametros:

        modelo = modelo_base(
            **parametros,
            random_state=42,
            verbosity=-1
        )

        resultado, modelo_entrenado = evaluar_modelo_clasificacion(
            modelo=modelo,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            nombre_modelo=nombre_modelo,
            estrategia_balance=estrategia_balance,
            parametros=parametros
        )

        resultados.append(resultado)

        auc_actual = resultado.loc[0, 'ROC-AUC']

        if auc_actual > mejor_auc:
            mejor_auc = auc_actual
            mejor_modelo = modelo_entrenado

    resultados = (
        pd.concat(resultados, ignore_index=True)
        .sort_values('ROC-AUC', ascending=False)
        .reset_index(drop=True)
    )

    return resultados, mejor_modelo

#=============================================================
def roc_plot_enhanced(model, features_valid, target_valid, model_name="Modelo"):
    '''
    Función que evalúa un modelo y muestra métricas completas junto con gráficas ROC.
    
    Parámetros:
    - model: modelo entrenado (LogisticRegression, RandomForest, etc.)
    - features_valid: características del conjunto de validación
    - target_valid: etiquetas verdaderas del conjunto de validación  
    - model_name: nombre del modelo para personalizar los títulos
    '''
    
    # Hacer predicciones
    valid_preds = model.predict(features_valid)
    probabilities_one_valid = model.predict_proba(features_valid)[:, 1]
    
    # Reporte de clasificación
    print(f"Reporte de clasificación en validación ({model_name}):")
    print(classification_report(target_valid, valid_preds))
    
    # Calcular métricas
    f1_score_val = f1_score(target_valid, valid_preds)
    auc_score = roc_auc_score(target_valid, probabilities_one_valid)
    
    print(f"F1 Score ({model_name}): {f1_score_val:.4f}")
    print(f"AUC-ROC ({model_name}): {auc_score:.4f}")
    
    # Calcular curvas para las gráficas
    fpr, tpr, thresholds_roc = roc_curve(target_valid, probabilities_one_valid)
    precision, recall, thresholds_pr = precision_recall_curve(target_valid, probabilities_one_valid)

    # Gráfica de recall vs precisión
    plt.figure(figsize=(6, 6))
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.show()

    #Prueba con diferentes umbrales
    thresholds_test = np.arange(0.0, 0.5, 0.05)
    results = []
    for threshold in thresholds_test:
        # Hacer predicciones con el nuevo umbral
        preds_threshold = (probabilities_one_valid >= threshold).astype(int)
        
        # Calcular métricas
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_valid, preds_threshold, average='binary'
        )
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    # Mostrar resultados
    results_df = pd.DataFrame(results)
    print(results_df)

    # Gráfica ROC
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title(f'Curva ROC ({model_name})')
    plt.show()
#============================================
def upsample_array(features, target, repeat):
    """
    Realiza oversampling de la clase minoritaria
    sobre matrices NumPy.

    Parameters
    ----------
    features : numpy.ndarray
        Matriz de características.

    target : array-like
        Variable objetivo.

    repeat : int
        Número de veces que se repite la clase minoritaria.

    Returns
    -------
    features_upsampled : numpy.ndarray
        Características después del oversampling.

    target_upsampled : numpy.ndarray
        Variable objetivo después del oversampling.
    """

    target = np.asarray(target)

    features_zeros = features[target == 0]
    features_ones = features[target == 1]

    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = np.concatenate(
        [features_zeros] + [features_ones] * repeat,
        axis=0
    )

    target_upsampled = np.concatenate(
        [target_zeros] + [target_ones] * repeat,
        axis=0
    )

    features_upsampled, target_upsampled = shuffle(
        features_upsampled,
        target_upsampled,
        random_state=54321
    )

    return features_upsampled, target_upsampled
#===============================================


#============================================================
import sklearn.metrics as metrics
def evaluate_model(model, train_features, train_target, test_features, test_target):

    eval_stats = {}

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):

        eval_stats[type] = {}

        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]

        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]

        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps

        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Valor F1
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Valor F1')

        # ROC
        ax = axs[1]
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')
        ax.set_title(f'Curva ROC')

        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)

    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1', 'APS', 'ROC AUC'))

    print(df_eval_stats)

    return