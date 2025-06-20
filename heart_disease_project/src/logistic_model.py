import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Importar las siguientes métricas
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib

def train_logistic_model(df, target='target_bin'):
    # Definir columnas a eliminar dinámicamente
    drop_candidates = [target, 'num']
    cols_to_drop = [col for col in drop_candidates if col in df.columns]

    # Variables predictoras
    features = df.drop(columns=cols_to_drop).columns.tolist()
    X = df[features]
    y = df[target]

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamiento del modelo de regresión logística
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # GUARDAR el modelo entrenado para uso posterior
    joblib.dump(model, './models/regresion_logistica_target_bin.pkl')

    # Predicción
    y_pred = model.predict(X_test)

    # Evaluación básica
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nEvaluación del modelo de regresión logística:')
    print(f'Accuracy: {accuracy:.2f}')

                                        # ⬅️Bloque Nuevo desde Aquí
    # -------------------------------
    # MÉTRICAS DE CLASIFICACIÓN
    # -------------------------------

    # Reporte de métricas
    print('\nReporte de Clasificación:')
    print(classification_report(y_test, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Matriz de Confusión - Regresión Logística')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Curva ROC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - Regresión Logística')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
                                        # ⬅️Bloque Nuevo Hasta Aquí



    # Visualización simple de predicción vs verdad
    # Compañeros, si lo desean pueden eliminar este gráfico y
    # conservar solo el de la matriz de confusión
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='mediumseagreen')
    plt.xlabel('Valor Real')
    plt.ylabel('Predicción')
    plt.title('Regresión Logística: Clasificación binaria')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
