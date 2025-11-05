import pickle
import pandas as pd
import numpy as np

print("="*80)
print("PROBADOR DE FLUJO REAL - MODELO MEJORADO")
print("="*80)

# ============================================================================
# 1. CARGAR MODELO MEJORADO
# ============================================================================
print("\n[1/2] Cargando modelo mejorado...")
try:
    with open('modelo_random_forest_mejorado.pkl', 'rb') as f:
        modelo_data = pickle.load(f)
    
    rf_model = modelo_data['modelo']
    estacion_map = modelo_data['estacion_map']
    features = modelo_data['features']
    periodo_map = modelo_data['periodo_map']
    
    print("✓ Modelo mejorado cargado exitosamente")
    print(f"✓ Estaciones disponibles: {len(estacion_map)}")
except FileNotFoundError:
    print("✗ Error: No se encontró 'modelo_random_forest_mejorado.pkl'")
    print("  Ejecuta primero el script de entrenamiento mejorado")
    exit()

# ============================================================================
# 2. FUNCIONES DE PREDICCIÓN MEJORADAS
# ============================================================================
def predecir_flujo(estacion_nombre, hora, dia_semana, es_fin_semana=False, minuto=0):
    """
    Predice el flujo REAL de pasajeros (con factor de expansión)
    
    Retorna: flujo en 30 minutos. Multiplica x2 para obtener flujo por hora.
    """
    
    if estacion_nombre not in estacion_map:
        print(f"❌ Error: Estación '{estacion_nombre}' no encontrada")
        return None
    
    if not 0 <= hora <= 23:
        print(f"❌ Error: Hora debe estar entre 0 y 23")
        return None
    
    if not 0 <= dia_semana <= 6:
        print(f"❌ Error: dia_semana debe estar entre 0 (Lunes) y 6 (Domingo)")
        return None
    
    # Clasificar periodo
    if 7 <= hora <= 9:
        periodo = 'punta_manana'
        periodo_encoded = periodo_map['punta_manana']
    elif 18 <= hora <= 20:
        periodo = 'punta_tarde'
        periodo_encoded = periodo_map['punta_tarde']
    elif 12 <= hora <= 14:
        periodo = 'mediodia'
        periodo_encoded = periodo_map['mediodia']
    else:
        periodo = 'valle'
        periodo_encoded = periodo_map['valle']
    
    # Calcular bloque de 30 minutos
    bloque_30min = hora * 2 + (minuto // 30)
    bloque_normalizado = bloque_30min / 48.0
    
    # Preparar datos
    estacion_encoded = estacion_map[estacion_nombre]
    es_hora_punta = 1 if periodo in ['punta_manana', 'punta_tarde'] else 0
    
    X_pred = pd.DataFrame({
        'estacion_encoded': [estacion_encoded],
        'hora': [hora],
        'bloque_30min': [bloque_30min],
        'dia_semana': [dia_semana],
        'periodo_encoded': [periodo_encoded],
        'es_fin_semana': [1 if es_fin_semana else 0],
        'es_hora_punta': [es_hora_punta]
    })
    
    # Hacer predicción
    flujo_predicho_30min = rf_model.predict(X_pred)[0]
    flujo_predicho_hora = flujo_predicho_30min * 2  # Extrapolación a 1 hora
    
    # Información contextual
    dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    dia_texto = dias[dia_semana]
    
    resultado = {
        'estacion': estacion_nombre,
        'hora': hora,
        'minuto': minuto,
        'dia': dia_texto,
        'periodo': periodo,
        'es_fin_semana': es_fin_semana,
        'flujo_30min': round(flujo_predicho_30min, 0),
        'flujo_hora': round(flujo_predicho_hora, 0)
    }
    
    return resultado

def mostrar_estaciones():
    """Muestra todas las estaciones disponibles"""
    estaciones = sorted(estacion_map.keys())
    print(f"\n📍 ESTACIONES DISPONIBLES ({len(estaciones)}):")
    print("-" * 80)
    for i, estacion in enumerate(estaciones, 1):
        print(f"{i:3d}. {estacion}")
    print("-" * 80)

def buscar_estacion(palabra):
    """Busca estaciones que contengan una palabra"""
    estaciones = [e for e in estacion_map.keys() if palabra.upper() in e.upper()]
    if estaciones:
        print(f"\n🔍 Estaciones que contienen '{palabra}':")
        for e in sorted(estaciones):
            print(f"   - {e}")
    else:
        print(f"❌ No se encontraron estaciones con '{palabra}'")

def comparar_periodos(estacion_nombre, dia_semana=0):
    """Compara el flujo en diferentes periodos del día"""
    periodos = [
        (6, "Madrugada (6 AM)"),
        (8, "Punta Mañana (8 AM)"),
        (12, "Mediodía (12 PM)"),
        (15, "Valle Tarde (3 PM)"),
        (19, "Punta Tarde (7 PM)"),
        (22, "Noche (10 PM)")
    ]
    
    dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    es_finde = dia_semana >= 5
    
    print(f"\n📊 COMPARACIÓN DE PERIODOS - {estacion_nombre}")
    print(f"   Día: {dias[dia_semana]}")
    print("-" * 80)
    print(f"{'Periodo':30s} {'Flujo/30min':>15s} {'Flujo/Hora':>15s}")
    print("-" * 80)
    
    for hora, nombre in periodos:
        resultado = predecir_flujo(estacion_nombre, hora, dia_semana, es_finde)
        if resultado:
            print(f"{nombre:30s} {resultado['flujo_30min']:>12,.0f} pax {resultado['flujo_hora']:>12,.0f} pax")
    print("-" * 80)

def comparar_dias(estacion_nombre, hora=8):
    """Compara el flujo en diferentes días de la semana"""
    dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    print(f"\n📊 COMPARACIÓN POR DÍA - {estacion_nombre}")
    print(f"   Hora: {hora}:00")
    print("-" * 80)
    print(f"{'Día':20s} {'Flujo/30min':>15s} {'Flujo/Hora':>15s}")
    print("-" * 80)
    
    for dia_num, dia_nombre in enumerate(dias):
        es_finde = dia_num >= 5
        resultado = predecir_flujo(estacion_nombre, hora, dia_num, es_finde)
        if resultado:
            print(f"{dia_nombre:20s} {resultado['flujo_30min']:>12,.0f} pax {resultado['flujo_hora']:>12,.0f} pax")
    print("-" * 80)

def analisis_completo_estacion(estacion_nombre):
    """Análisis completo de una estación"""
    print(f"\n" + "="*80)
    print(f"ANÁLISIS COMPLETO: {estacion_nombre}")
    print("="*80)
    
    # Día laboral típico
    print("\n🔷 LUNES TÍPICO:")
    comparar_periodos(estacion_nombre, dia_semana=0)
    
    # Fin de semana
    print("\n🔷 SÁBADO TÍPICO:")
    comparar_periodos(estacion_nombre, dia_semana=5)
    
    # Hora punta por día
    print("\n🔷 HORA PUNTA MAÑANA (8 AM) - TODA LA SEMANA:")
    comparar_dias(estacion_nombre, hora=8)

# ============================================================================
# 3. EJEMPLOS REALISTAS
# ============================================================================
print("\n[2/2] Ejecutando predicciones realistas...")

# Seleccionar estación de prueba
estacion_test = sorted(estacion_map.keys())[0]

print("\n" + "="*80)
print("EJEMPLOS DE PREDICCIONES REALISTAS")
print("="*80)

# Ejemplo 1: Hora punta mañana - Lunes
print(f"\n🔷 EJEMPLO 1: HORA PUNTA MAÑANA (ALTA DEMANDA)")
r1 = predecir_flujo(estacion_test, hora=8, dia_semana=0)
if r1:
    print(f"   🚇 Estación: {r1['estacion']}")
    print(f"   📅 Día: {r1['dia']} (Laboral)")
    print(f"   🕐 Hora: {r1['hora']}:00")
    print(f"   ⏰ Periodo: {r1['periodo']}")
    print(f"   ➡️  Flujo en 30 min: {r1['flujo_30min']:,.0f} pasajeros")
    print(f"   ➡️  Flujo estimado/hora: {r1['flujo_hora']:,.0f} pasajeros")

# Ejemplo 2: Valle
print(f"\n🔷 EJEMPLO 2: HORA VALLE (BAJA DEMANDA)")
r2 = predecir_flujo(estacion_test, hora=15, dia_semana=2)
if r2:
    print(f"   🚇 Estación: {r2['estacion']}")
    print(f"   📅 Día: {r2['dia']} (Laboral)")
    print(f"   🕐 Hora: {r2['hora']}:00")
    print(f"   ⏰ Periodo: {r2['periodo']}")
    print(f"   ➡️  Flujo en 30 min: {r2['flujo_30min']:,.0f} pasajeros")
    print(f"   ➡️  Flujo estimado/hora: {r2['flujo_hora']:,.0f} pasajeros")

# Ejemplo 3: Hora punta tarde - Viernes
print(f"\n🔷 EJEMPLO 3: HORA PUNTA TARDE (ALTA DEMANDA)")
r3 = predecir_flujo(estacion_test, hora=19, dia_semana=4)
if r3:
    print(f"   🚇 Estación: {r3['estacion']}")
    print(f"   📅 Día: {r3['dia']} (Laboral)")
    print(f"   🕐 Hora: {r3['hora']}:00")
    print(f"   ⏰ Periodo: {r3['periodo']}")
    print(f"   ➡️  Flujo en 30 min: {r3['flujo_30min']:,.0f} pasajeros")
    print(f"   ➡️  Flujo estimado/hora: {r3['flujo_hora']:,.0f} pasajeros")

# Comparación completa
print("\n" + "="*80)
print("COMPARACIÓN DETALLADA")
print("="*80)
comparar_periodos(estacion_test, dia_semana=0)

# ============================================================================
# 4. GUÍA DE USO
# ============================================================================
print("\n" + "="*80)
print("📖 GUÍA DE USO RÁPIDA")
print("="*80)

print("""
🔹 PREDICCIÓN SIMPLE:
   resultado = predecir_flujo('BAQUEDANO', hora=8, dia_semana=0)
   print(f"Flujo/hora: {resultado['flujo_hora']:,.0f} pasajeros")

🔹 VER ESTACIONES:
   mostrar_estaciones()

🔹 BUSCAR ESTACIÓN:
   buscar_estacion('TOBALABA')

🔹 COMPARAR PERIODOS DEL DÍA:
   comparar_periodos('BAQUEDANO', dia_semana=0)

🔹 COMPARAR DÍAS DE LA SEMANA:
   comparar_dias('BAQUEDANO', hora=8)

🔹 ANÁLISIS COMPLETO:
   analisis_completo_estacion('BAQUEDANO')

📌 NOTA: Las predicciones son para bloques de 30 minutos.
         Multiplica x2 para obtener flujo estimado por hora.
""")

print("\n✅ Modelo listo para usar con predicciones realistas")
print("="*80)