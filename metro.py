import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("FILTRACIÓN DE DATOS - METRO DE SANTIAGO")
print("Fundamentos de Data Science - UTEM")
print("="*70)

# 1. CARGAR DATOS
print("\n[1/6] Cargando datos del CSV...")
try:
    # Cargar con separador pipe
    df = pd.read_csv('metro.csv', sep='|', low_memory=False)
    print(f"✓ Datos cargados: {df.shape[0]:,} registros, {df.shape[1]} columnas")
except Exception as e:
    print(f"✗ Error al cargar archivo: {e}")
    exit()

# 2. EXPLORACIÓN INICIAL
print("\n[2/6] Explorando estructura de datos...")
print(f"\nPrimeras columnas: {list(df.columns[:10])}")
print(f"\nTipos de transporte únicos en tipo_transporte_1:")
print(df['tipo_transporte_1'].value_counts())

print(f"\nOperadores únicos en op_1era_etapa (primeros 10):")
print(df['op_1era_etapa'].value_counts().head(10))

# 3. IDENTIFICAR REGISTROS DE METRO
print("\n[3/6] Identificando registros de Metro...")

# Criterios para filtrar SOLO Metro:
# - tipo_transporte_1 debe ser 1 (Metro) o 2 si es combinado pero inicia en metro
# - op_1era_etapa debe ser 1 (operador Metro de Santiago)
# - Debe tener información de estaciones, no paraderos de bus

# Filtro principal: operador = 1 (Metro)
df_metro = df[df['op_1era_etapa'] == 1].copy()

print(f"✓ Registros con operador Metro (op_1era_etapa = 1): {len(df_metro):,}")

# Verificar si hay columnas de estaciones de metro
cols_estaciones = [col for col in df.columns if 'paradero_subida' in col or 'paradero_bajada' in col]
print(f"\nColumnas de estaciones encontradas: {cols_estaciones[:4]}")

# 4. LIMPIEZA DE DATOS
print("\n[4/6] Limpiando datos de Metro...")

# Eliminar registros sin bajada (viajes incompletos)
print(f"Registros antes de limpiar: {len(df_metro):,}")

# Filtrar registros con información completa de viaje
df_metro_clean = df_metro[
    (df_metro['netapassinbajada'] == 0) &  # Tiene bajada registrada
    (df_metro['ultimaetapaconbajada'] == 1)  # Última etapa con bajada
].copy()

print(f"✓ Registros después de eliminar viajes sin bajada: {len(df_metro_clean):,}")

# Eliminar columnas irrelevantes (buses)
cols_to_drop = [col for col in df_metro_clean.columns if 'srv_' in col or 'dveh' in col]
df_metro_clean = df_metro_clean.drop(columns=cols_to_drop, errors='ignore')

print(f"✓ Columnas eliminadas: {len(cols_to_drop)}")
print(f"✓ Columnas restantes: {df_metro_clean.shape[1]}")

# 5. ESTADÍSTICAS DESCRIPTIVAS
print("\n[5/6] Estadísticas de datos filtrados...")
print(f"\nRegistros totales de Metro: {len(df_metro_clean):,}")
print(f"Periodo de datos: {df_metro_clean.columns[0] if 'fecha' in str(df_metro_clean.columns) else 'Verificar columnas de fecha'}")

# Verificar valores nulos
print(f"\nColumnas con valores nulos:")
null_counts = df_metro_clean.isnull().sum()
print(null_counts[null_counts > 0].head(10))

# Estadísticas de tipos de transporte en las etapas
print(f"\nDistribución de tipo_transporte_1 en datos filtrados:")
print(df_metro_clean['tipo_transporte_1'].value_counts())

# 6. GUARDAR DATOS LIMPIOS
print("\n[6/6] Guardando datos filtrados...")
output_file = 'metro_filtrado.csv'
df_metro_clean.to_csv(output_file, index=False, sep='|')
print(f"✓ Archivo guardado: {output_file}")
print(f"✓ Tamaño del archivo: {len(df_metro_clean):,} registros")

# RESUMEN FINAL
print("\n" + "="*70)
print("RESUMEN DE FILTRACIÓN")
print("="*70)
print(f"Registros originales:        {len(df):,}")
print(f"Registros de Metro:          {len(df_metro):,}")
print(f"Registros limpios (finales): {len(df_metro_clean):,}")
print(f"Reducción:                   {(1 - len(df_metro_clean)/len(df))*100:.1f}%")
print("="*70)

# VISUALIZACIÓN DE DATOS FILTRADOS
print("\n[BONUS] Generando visualizaciones iniciales...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis Exploratorio Inicial - Metro de Santiago', fontsize=16, fontweight='bold')

# Gráfico 1: Distribución de tipos de transporte
ax1 = axes[0, 0]
tipo_counts = df_metro_clean['tipo_transporte_1'].value_counts()
ax1.bar(tipo_counts.index.astype(str), tipo_counts.values, color='steelblue', alpha=0.7)
ax1.set_xlabel('Tipo de Transporte', fontweight='bold')
ax1.set_ylabel('Cantidad de Viajes', fontweight='bold')
ax1.set_title('Distribución por Tipo de Transporte')
ax1.grid(axis='y', alpha=0.3)

# Gráfico 2: Viajes con/sin bajada
ax2 = axes[0, 1]
bajada_counts = df_metro_clean['netapassinbajada'].value_counts()
# Ajustar labels según los datos disponibles
if len(bajada_counts) == 1:
    labels = ['Con Bajada'] if bajada_counts.index[0] == 0 else ['Sin Bajada']
    colors = ['#2ecc71']
else:
    labels = ['Con Bajada', 'Sin Bajada']
    colors = ['#2ecc71', '#e74c3c']
ax2.pie(bajada_counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('Completitud de Viajes')

# Gráfico 3: Top 10 columnas con valores nulos
ax3 = axes[1, 0]
top_nulls = df_metro_clean.isnull().sum().sort_values(ascending=False).head(10)
if len(top_nulls) > 0:
    ax3.barh(range(len(top_nulls)), top_nulls.values, color='coral', alpha=0.7)
    ax3.set_yticks(range(len(top_nulls)))
    ax3.set_yticklabels([col[:20] + '...' if len(col) > 20 else col for col in top_nulls.index])
    ax3.set_xlabel('Cantidad de Nulos', fontweight='bold')
    ax3.set_title('Top 10 Columnas con Valores Nulos')
    ax3.grid(axis='x', alpha=0.3)
else:
    ax3.text(0.5, 0.5, '¡Sin valores nulos!', ha='center', va='center', fontsize=14)
    ax3.axis('off')

# Gráfico 4: Información del dataset
ax4 = axes[1, 1]
ax4.axis('off')
info_text = f"""
INFORMACIÓN DEL DATASET FILTRADO

Registros totales:     {len(df_metro_clean):,}
Columnas:             {df_metro_clean.shape[1]}
Memoria (MB):         {df_metro_clean.memory_usage(deep=True).sum() / 1024**2:.2f}

Operador principal:   Metro de Santiago (op=1)
Viajes completos:     100%
Valores nulos:        {df_metro_clean.isnull().sum().sum():,}

Estado: ✓ LISTO PARA ANÁLISIS
"""
ax4.text(0.1, 0.5, info_text, fontsize=11, family='monospace', 
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('exploracion_inicial_metro.png', dpi=300, bbox_inches='tight')
print("✓ Visualización guardada: exploracion_inicial_metro.png")

print("\n✓ FILTRACIÓN COMPLETADA EXITOSAMENTE")
print(f"Próximo paso: Análisis exploratorio y preparación de variables para modelado")