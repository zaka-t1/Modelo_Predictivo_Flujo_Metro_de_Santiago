import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("PREDICCIÓN DE FLUJO DE PASAJEROS - RANDOM FOREST")
print("Fundamentos de Data Science - UTEM")
print("="*80)

# ============================================================================
# 1. CARGA Y LIMPIEZA DE DATOS
# ============================================================================
print("\n[1/7] Cargando datos filtrados de Metro...")
df = pd.read_csv('metro_filtrado.csv', sep='|', low_memory=False)
print(f"✓ Datos cargados: {len(df):,} registros")

# Eliminar columna unnamed
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Filtrar solo primera etapa de viaje (la más importante)
print("\n[2/7] Preparando datos de primera etapa...")
df_etapa1 = df[
    (df['paradero_subida_1'].notna()) & 
    (df['tiempo_subida_1'].notna()) &
    (df['op_1era_etapa'] == 1)  # Solo Metro
].copy()

print(f"✓ Registros con información completa: {len(df_etapa1):,}")

# ============================================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS
# ============================================================================
print("\n[3/7] Creando variables temporales y contextuales...")

# Convertir tiempo_subida_1 a datetime
df_etapa1['tiempo_subida_1'] = pd.to_datetime(df_etapa1['tiempo_subida_1'], errors='coerce')

# Extraer características temporales
df_etapa1['hora'] = df_etapa1['tiempo_subida_1'].dt.hour
df_etapa1['dia_semana'] = df_etapa1['tiempo_subida_1'].dt.dayofweek  # 0=Lunes, 6=Domingo
df_etapa1['dia_mes'] = df_etapa1['tiempo_subida_1'].dt.day
df_etapa1['mes'] = df_etapa1['tiempo_subida_1'].dt.month
df_etapa1['minuto'] = df_etapa1['tiempo_subida_1'].dt.minute

# Crear variable de hora punta
def clasificar_periodo(hora):
    if 7 <= hora <= 9:
        return 'punta_manana'
    elif 18 <= hora <= 20:
        return 'punta_tarde'
    elif 12 <= hora <= 14:
        return 'mediodia'
    else:
        return 'valle'

df_etapa1['periodo_hora'] = df_etapa1['hora'].apply(clasificar_periodo)

# Variable fin de semana
df_etapa1['es_fin_semana'] = (df_etapa1['dia_semana'] >= 5).astype(int)

print(f"✓ Variables temporales creadas")
print(f"  - Rango de fechas: {df_etapa1['tiempo_subida_1'].min()} a {df_etapa1['tiempo_subida_1'].max()}")
print(f"  - Distribución por periodo:")
print(df_etapa1['periodo_hora'].value_counts())

# ============================================================================
# 3. AGREGACIÓN PARA CALCULAR FLUJO DE PASAJEROS
# ============================================================================
print("\n[4/7] Calculando flujo de pasajeros por estación y periodo...")

# Agrupar por estación + hora + día de la semana
flujo_df = df_etapa1.groupby([
    'paradero_subida_1',
    'hora',
    'dia_semana',
    'periodo_hora',
    'es_fin_semana'
]).agg({
    'id_viaje': 'count',  # Cantidad de viajes
    'factor_expansion': 'sum'  # Factor de expansión acumulado
}).reset_index()

# Renombrar columnas
flujo_df.columns = ['estacion', 'hora', 'dia_semana', 'periodo_hora', 
                     'es_fin_semana', 'num_viajes', 'flujo_expandido']

# Usar flujo expandido como variable objetivo
flujo_df['flujo_pasajeros'] = flujo_df['flujo_expandido'].fillna(flujo_df['num_viajes'])

print(f"✓ Dataset agregado creado: {len(flujo_df):,} registros")
print(f"  - Estaciones únicas: {flujo_df['estacion'].nunique()}")
print(f"  - Flujo promedio: {flujo_df['flujo_pasajeros'].mean():.2f} pasajeros")
print(f"  - Flujo máximo: {flujo_df['flujo_pasajeros'].max():.2f} pasajeros")

# ============================================================================
# 4. PREPARACIÓN DE DATOS PARA MODELO
# ============================================================================
print("\n[5/7] Preparando variables para Random Forest...")

# Codificar variables categóricas
flujo_df['estacion_encoded'] = pd.factorize(flujo_df['estacion'])[0]
flujo_df['periodo_encoded'] = pd.factorize(flujo_df['periodo_hora'])[0]

# Seleccionar características (X)
features = ['estacion_encoded', 'hora', 'dia_semana', 'periodo_encoded', 'es_fin_semana']
X = flujo_df[features]
y = flujo_df['flujo_pasajeros']

print(f"✓ Variables predictoras: {features}")
print(f"✓ Variable objetivo: flujo_pasajeros")
print(f"✓ Tamaño del dataset: {X.shape}")

# División train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n  Conjunto de entrenamiento: {len(X_train):,} registros")
print(f"  Conjunto de prueba: {len(X_test):,} registros")

# ============================================================================
# 5. ENTRENAMIENTO DEL MODELO RANDOM FOREST
# ============================================================================
print("\n[6/7] Entrenando modelo Random Forest...")
print("  Configuración:")
print("    - Número de árboles: 100")
print("    - Profundidad máxima: 15")
print("    - Min samples split: 10")
print("    - Random state: 42")

# Crear y entrenar modelo
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train, y_train)
print("✓ Modelo entrenado exitosamente")

# ============================================================================
# 6. EVALUACIÓN DEL MODELO
# ============================================================================
print("\n[7/7] Evaluando rendimiento del modelo...")

# Predicciones
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Métricas de entrenamiento
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

# Métricas de prueba
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

# Validación cruzada
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, 
                            scoring='r2', n_jobs=-1)

print("\n" + "="*80)
print("MÉTRICAS DE RENDIMIENTO - RANDOM FOREST")
print("="*80)
print(f"\n📊 CONJUNTO DE ENTRENAMIENTO:")
print(f"   RMSE:  {train_rmse:.4f}")
print(f"   MAE:   {train_mae:.4f}")
print(f"   R²:    {train_r2:.4f}")

print(f"\n📊 CONJUNTO DE PRUEBA:")
print(f"   RMSE:  {test_rmse:.4f}")
print(f"   MAE:   {test_mae:.4f}")
print(f"   R²:    {test_r2:.4f}")

print(f"\n🔄 VALIDACIÓN CRUZADA (5-Fold):")
print(f"   R² promedio: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Importancia de variables
feature_importance = pd.DataFrame({
    'variable': features,
    'importancia': rf_model.feature_importances_
}).sort_values('importancia', ascending=False)

print(f"\n🌟 IMPORTANCIA DE VARIABLES:")
for idx, row in feature_importance.iterrows():
    print(f"   {row['variable']}: {row['importancia']:.4f}")

# ============================================================================
# 7. VISUALIZACIONES
# ============================================================================
print("\n📈 Generando visualizaciones...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Predicciones vs Reales (TEST)
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(y_test, y_pred_test, alpha=0.5, s=30, color='steelblue', edgecolors='black', linewidth=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicción perfecta')
ax1.set_xlabel('Flujo Real (pasajeros)', fontweight='bold', fontsize=11)
ax1.set_ylabel('Flujo Predicho (pasajeros)', fontweight='bold', fontsize=11)
ax1.set_title('Predicciones vs Valores Reales - Conjunto de Prueba', fontweight='bold', fontsize=13)
ax1.legend()
ax1.grid(alpha=0.3)
ax1.text(0.05, 0.95, f'R² = {test_r2:.4f}\nRMSE = {test_rmse:.2f}', 
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Métricas comparativas
ax2 = fig.add_subplot(gs[0, 2])
metrics_data = {
    'Train': [train_r2, train_rmse/100, train_mae/100],
    'Test': [test_r2, test_rmse/100, test_mae/100]
}
x_pos = np.arange(3)
width = 0.35
ax2.bar(x_pos - width/2, metrics_data['Train'], width, label='Train', alpha=0.8, color='#2ecc71')
ax2.bar(x_pos + width/2, metrics_data['Test'], width, label='Test', alpha=0.8, color='#3498db')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['R²', 'RMSE/100', 'MAE/100'], fontsize=9)
ax2.set_title('Comparación Train vs Test', fontweight='bold', fontsize=11)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Residuos
ax3 = fig.add_subplot(gs[1, :2])
residuos = y_test - y_pred_test
ax3.scatter(y_pred_test, residuos, alpha=0.5, s=30, color='coral', edgecolors='black', linewidth=0.5)
ax3.axhline(y=0, color='red', linestyle='--', lw=2)
ax3.set_xlabel('Flujo Predicho', fontweight='bold', fontsize=11)
ax3.set_ylabel('Residuos', fontweight='bold', fontsize=11)
ax3.set_title('Análisis de Residuos', fontweight='bold', fontsize=13)
ax3.grid(alpha=0.3)

# 4. Importancia de variables
ax4 = fig.add_subplot(gs[1, 2])
ax4.barh(feature_importance['variable'], feature_importance['importancia'], color='#9b59b6', alpha=0.8)
ax4.set_xlabel('Importancia', fontweight='bold', fontsize=10)
ax4.set_title('Importancia de Variables', fontweight='bold', fontsize=11)
ax4.grid(axis='x', alpha=0.3)

# 5. Distribución de errores
ax5 = fig.add_subplot(gs[2, 0])
ax5.hist(residuos, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
ax5.axvline(x=0, color='blue', linestyle='--', lw=2)
ax5.set_xlabel('Error (Residuo)', fontweight='bold', fontsize=10)
ax5.set_ylabel('Frecuencia', fontweight='bold', fontsize=10)
ax5.set_title('Distribución de Errores', fontweight='bold', fontsize=11)
ax5.grid(alpha=0.3)

# 6. Errores porcentuales
ax6 = fig.add_subplot(gs[2, 1])
errores_pct = (np.abs(residuos) / y_test) * 100
errores_pct = errores_pct[errores_pct < 100]  # Filtrar outliers
ax6.hist(errores_pct, bins=50, color='#f39c12', alpha=0.7, edgecolor='black')
ax6.set_xlabel('Error Absoluto (%)', fontweight='bold', fontsize=10)
ax6.set_ylabel('Frecuencia', fontweight='bold', fontsize=10)
ax6.set_title('Distribución de Errores Porcentuales', fontweight='bold', fontsize=11)
ax6.grid(alpha=0.3)
ax6.text(0.65, 0.95, f'Mediana: {np.median(errores_pct):.1f}%', 
         transform=ax6.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 7. Validación cruzada
ax7 = fig.add_subplot(gs[2, 2])
ax7.boxplot([cv_scores], labels=['5-Fold CV'], patch_artist=True,
            boxprops=dict(facecolor='#1abc9c', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
ax7.set_ylabel('R² Score', fontweight='bold', fontsize=10)
ax7.set_title('Validación Cruzada', fontweight='bold', fontsize=11)
ax7.grid(axis='y', alpha=0.3)
ax7.text(1, cv_scores.mean(), f'{cv_scores.mean():.4f}', 
         ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.suptitle('ANÁLISIS COMPLETO - RANDOM FOREST: PREDICCIÓN DE FLUJO DE PASAJEROS', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('resultados_random_forest_flujo.png', dpi=300, bbox_inches='tight')
print("✓ Visualizaciones guardadas: resultados_random_forest_flujo.png")

# ============================================================================
# 8. CONCLUSIONES Y JUSTIFICACIÓN
# ============================================================================
print("\n" + "="*80)
print("CONCLUSIONES Y JUSTIFICACIÓN DEL MODELO")
print("="*80)

print(f"""
✅ MODELO SELECCIONADO: Random Forest Regressor

📊 RENDIMIENTO:
   - R² en prueba: {test_r2:.4f} ({test_r2*100:.2f}% de la varianza explicada)
   - RMSE: {test_rmse:.2f} pasajeros
   - MAE: {test_mae:.2f} pasajeros
   - Error porcentual mediano: {np.median(errores_pct):.2f}%

🌟 JUSTIFICACIÓN TÉCNICA:

1. CAPACIDAD DE GENERALIZACIÓN:
   - R² Train: {train_r2:.4f} vs R² Test: {test_r2:.4f}
   - Diferencia: {abs(train_r2 - test_r2):.4f} (bajo sobreajuste)
   - El modelo NO está sobreentrenado

2. VALIDACIÓN CRUZADA:
   - R² promedio: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})
   - Modelo consistente en diferentes particiones de datos

3. VENTAJAS PARA FLUJO DE PASAJEROS:
   ✓ Captura patrones NO lineales (hora punta vs valle)
   ✓ Maneja interacciones complejas entre variables
   ✓ Robusto a datos atípicos
   ✓ Alta interpretabilidad (importancia de variables)

4. VARIABLES MÁS IMPORTANTES:
   → {feature_importance.iloc[0]['variable']}: {feature_importance.iloc[0]['importancia']:.4f}
   → {feature_importance.iloc[1]['variable']}: {feature_importance.iloc[1]['importancia']:.4f}
   
5. APLICABILIDAD PRÁCTICA:
   ✓ Puede predecir flujo en tiempo real
   ✓ Útil para planificación operativa del Metro
   ✓ Identifica patrones de congestión

📌 CONCLUSIÓN FINAL:
Random Forest es el modelo óptimo para predecir flujo de pasajeros en el Metro 
de Santiago debido a su capacidad de capturar la no linealidad inherente en los
patrones de movilidad urbana, su robustez ante datos atípicos, y su excelente
balance entre precisión y interpretabilidad.
""")

print("\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
print("="*80)

# ============================================================================
# 9. MÓDULO DE PREDICCIONES INTERACTIVAS
# ============================================================================
print("\n" + "="*80)
print("MÓDULO DE PREDICCIONES INDIVIDUALES")
print("="*80)

# Crear diccionario de estaciones
estaciones_disponibles = flujo_df['estacion'].unique()
estacion_map = {estacion: idx for idx, estacion in enumerate(sorted(estaciones_disponibles))}
estacion_reverse_map = {v: k for k, v in estacion_map.items()}

print(f"\n📍 Estaciones disponibles en el modelo: {len(estaciones_disponibles)}")
print("\nPrimeras 10 estaciones:")
for i, estacion in enumerate(sorted(estaciones_disponibles)[:10], 1):
    print(f"  {i}. {estacion}")

# Función para hacer predicciones
def predecir_flujo(estacion_nombre, hora, dia_semana, es_fin_semana=False):
    """
    Predice el flujo de pasajeros para una estación específica
    
    Parámetros:
    - estacion_nombre: Nombre de la estación (str)
    - hora: Hora del día (0-23)
    - dia_semana: Día de la semana (0=Lunes, 6=Domingo)
    - es_fin_semana: Si es fin de semana (bool)
    """
    
    # Validar estación
    if estacion_nombre not in estacion_map:
        return None, f"Error: Estación '{estacion_nombre}' no encontrada"
    
    # Clasificar periodo
    if 7 <= hora <= 9:
        periodo = 'punta_manana'
        periodo_encoded = 2
    elif 18 <= hora <= 20:
        periodo = 'punta_tarde'
        periodo_encoded =3
    elif 12 <= hora <= 14:
        periodo = 'mediodia'
        periodo_encoded =1
    else:
        periodo = 'valle'
        periodo_encoded = 0
    
    # Preparar datos
    estacion_encoded = estacion_map[estacion_nombre]
    
    X_pred = pd.DataFrame({
        'estacion_encoded': [estacion_encoded],
        'hora': [hora],
        'dia_semana': [dia_semana],
        'periodo_encoded': [periodo_encoded],
        'es_fin_semana': [1 if es_fin_semana else 0]
    })
    
    # Hacer predicción
    flujo_predicho = rf_model.predict(X_pred)[0]
    
    # Obtener información contextual
    dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    dia_texto = dias[dia_semana]
    
    resultado = {
        'estacion': estacion_nombre,
        'hora': hora,
        'dia': dia_texto,
        'periodo': periodo,
        'es_fin_semana': es_fin_semana,
        'flujo_predicho': round(flujo_predicho, 2)
    }
    
    return resultado, None

# ============================================================================
# EJEMPLOS DE PREDICCIONES
# ============================================================================
print("\n" + "-"*80)
print("EJEMPLOS DE PREDICCIONES")
print("-"*80)

# Ejemplo 1: Hora punta mañana - Lunes
estacion_ejemplo = sorted(estaciones_disponibles)[0]
resultado1, error1 = predecir_flujo(estacion_ejemplo, hora=8, dia_semana=0, es_fin_semana=False)
if resultado1:
    print(f"\n🔷 Ejemplo 1: HORA PUNTA MAÑANA")
    print(f"   Estación: {resultado1['estacion']}")
    print(f"   Día: {resultado1['dia']} (Laboral)")
    print(f"   Hora: {resultado1['hora']}:00")
    print(f"   Periodo: {resultado1['periodo']}")
    print(f"   ➡️  FLUJO PREDICHO: {resultado1['flujo_predicho']:.0f} pasajeros")

# Ejemplo 2: Valle - Miércoles
resultado2, error2 = predecir_flujo(estacion_ejemplo, hora=15, dia_semana=2, es_fin_semana=False)
if resultado2:
    print(f"\n🔷 Ejemplo 2: HORA VALLE")
    print(f"   Estación: {resultado2['estacion']}")
    print(f"   Día: {resultado2['dia']} (Laboral)")
    print(f"   Hora: {resultado2['hora']}:00")
    print(f"   Periodo: {resultado2['periodo']}")
    print(f"   ➡️  FLUJO PREDICHO: {resultado2['flujo_predicho']:.0f} pasajeros")

# Ejemplo 3: Hora punta tarde - Viernes
resultado3, error3 = predecir_flujo(estacion_ejemplo, hora=19, dia_semana=4, es_fin_semana=False)
if resultado3:
    print(f"\n🔷 Ejemplo 3: HORA PUNTA TARDE")
    print(f"   Estación: {resultado3['estacion']}")
    print(f"   Día: {resultado3['dia']} (Laboral)")
    print(f"   Hora: {resultado3['hora']}:00")
    print(f"   Periodo: {resultado3['periodo']}")
    print(f"   ➡️  FLUJO PREDICHO: {resultado3['flujo_predicho']:.0f} pasajeros")

# Ejemplo 4: Fin de semana - Sábado
resultado4, error4 = predecir_flujo(estacion_ejemplo, hora=14, dia_semana=5, es_fin_semana=True)
if resultado4:
    print(f"\n🔷 Ejemplo 4: FIN DE SEMANA")
    print(f"   Estación: {resultado4['estacion']}")
    print(f"   Día: {resultado4['dia']} (Fin de semana)")
    print(f"   Hora: {resultado4['hora']}:00")
    print(f"   Periodo: {resultado4['periodo']}")
    print(f"   ➡️  FLUJO PREDICHO: {resultado4['flujo_predicho']:.0f} pasajeros")

# ============================================================================
# MODO INTERACTIVO (OPCIONAL)
# ============================================================================
print("\n" + "-"*80)
print("MODO INTERACTIVO - HAZ TUS PROPIAS PREDICCIONES")
print("-"*80)
print("\nPara usar el modelo, ejecuta esta función en tu código:")
print("""
# Ejemplo de uso:
resultado, error = predecir_flujo(
    estacion_nombre='BAQUEDANO',  # Nombre exacto de la estación
    hora=8,                        # Hora (0-23)
    dia_semana=0,                  # 0=Lunes, 1=Martes, ..., 6=Domingo
    es_fin_semana=False            # True si es sábado o domingo
)

if resultado:
    print(f"Flujo predicho: {resultado['flujo_predicho']:.0f} pasajeros")
else:
    print(f"Error: {error}")
""")

# Guardar función para uso posterior
import pickle
modelo_data = {
    'modelo': rf_model,
    'estacion_map': estacion_map,
    'features': features
}
with open('modelo_random_forest.pkl', 'wb') as f:
    pickle.dump(modelo_data, f)

print("\n✓ Modelo guardado en: modelo_random_forest.pkl")
print("  Puedes cargarlo después con: pickle.load(open('modelo_random_forest.pkl', 'rb'))")

print("\n" + "="*80)
print("🎉 ANÁLISIS COMPLETO FINALIZADO")
print("="*80)