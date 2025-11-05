import pickle
import pandas as pd
import numpy as np

print("="*80)
print("PROBADOR DE PREDICCIONES - FLUJO DE PASAJEROS METRO")
print("="*80)

# ============================================================================
# 1. CARGAR MODELO ENTRENADO
# ============================================================================
print("\n[1/3] Cargando modelo entrenado...")
try:
    with open('modelo_random_forest.pkl', 'rb') as f:
        modelo_data = pickle.load(f)
    
    rf_model = modelo_data['modelo']
    estacion_map = modelo_data['estacion_map']
    features = modelo_data['features']
    
    print("✓ Modelo cargado exitosamente")
    print(f"✓ Estaciones disponibles: {len(estacion_map)}")
except FileNotFoundError:
    print("✗ Error: No se encontró 'modelo_random_forest.pkl'")
    print("  Primero ejecuta el script de entrenamiento principal")
    exit()

# ============================================================================
# 2. FUNCIÓN DE PREDICCIÓN
# ============================================================================
def predecir_flujo(estacion_nombre, hora, dia_semana, es_fin_semana=False):
    """
    Predice el flujo de pasajeros para una estación específica
    
    Parámetros:
    - estacion_nombre: Nombre de la estación (str)
    - hora: Hora del día (0-23)
    - dia_semana: Día de la semana (0=Lunes, 6=Domingo)
    - es_fin_semana: Si es fin de semana (bool)
    
    Retorna:
    - dict con la predicción o None si hay error
    """
    
    # Validar estación
    if estacion_nombre not in estacion_map:
        print(f"❌ Error: Estación '{estacion_nombre}' no encontrada")
        print(f"   Usa mostrar_estaciones() para ver las disponibles")
        return None
    
    # Validar hora
    if not 0 <= hora <= 23:
        print(f"❌ Error: Hora debe estar entre 0 y 23")
        return None
    
    # Validar día
    if not 0 <= dia_semana <= 6:
        print(f"❌ Error: dia_semana debe estar entre 0 (Lunes) y 6 (Domingo)")
        return None
    
    # Clasificar periodo
    if 7 <= hora <= 9:
        periodo = 'punta_manana'
        periodo_encoded = 2
    elif 18 <= hora <= 20:
        periodo = 'punta_tarde'
        periodo_encoded = 3
    elif 12 <= hora <= 14:
        periodo = 'mediodia'
        periodo_encoded = 1
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
    
    return resultado

# ============================================================================
# 3. FUNCIONES AUXILIARES
# ============================================================================
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
        (7, "Punta Mañana (7 AM)"),
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
    
    for hora, nombre in periodos:
        resultado = predecir_flujo(estacion_nombre, hora, dia_semana, es_finde)
        if resultado:
            print(f"   {nombre:25s} → {resultado['flujo_predicho']:>8.0f} pasajeros")
    print("-" * 80)

def comparar_dias(estacion_nombre, hora=8):
    """Compara el flujo en diferentes días de la semana"""
    dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    print(f"\n📊 COMPARACIÓN POR DÍA - {estacion_nombre}")
    print(f"   Hora: {hora}:00")
    print("-" * 80)
    
    for dia_num, dia_nombre in enumerate(dias):
        es_finde = dia_num >= 5
        resultado = predecir_flujo(estacion_nombre, hora, dia_num, es_finde)
        if resultado:
            print(f"   {dia_nombre:15s} → {resultado['flujo_predicho']:>8.0f} pasajeros")
    print("-" * 80)

# ============================================================================
# 4. EJEMPLOS DE USO
# ============================================================================
print("\n[2/3] Ejecutando ejemplos de predicción...")

# Ejemplo 1: Predicción simple
print("\n" + "="*80)
print("EJEMPLO 1: PREDICCIÓN SIMPLE")
print("="*80)

estacion_test = sorted(estacion_map.keys())[0]  # Primera estación disponible
resultado = predecir_flujo(
    estacion_nombre=estacion_test,
    hora=8,
    dia_semana=0,  # Lunes
    es_fin_semana=False
)

if resultado:
    print(f"\n✅ RESULTADO:")
    print(f"   🚇 Estación: {resultado['estacion']}")
    print(f"   📅 Día: {resultado['dia']}")
    print(f"   🕐 Hora: {resultado['hora']}:00")
    print(f"   ⏰ Periodo: {resultado['periodo']}")
    print(f"   👥 FLUJO PREDICHO: {resultado['flujo_predicho']:.0f} pasajeros")

# Ejemplo 2: Comparación de periodos
print("\n" + "="*80)
print("EJEMPLO 2: COMPARACIÓN DE PERIODOS DEL DÍA")
print("="*80)
comparar_periodos(estacion_test, dia_semana=0)

# Ejemplo 3: Comparación por día de la semana
print("\n" + "="*80)
print("EJEMPLO 3: COMPARACIÓN POR DÍA DE LA SEMANA")
print("="*80)
comparar_dias(estacion_test, hora=8)

# ============================================================================
# 5. INSTRUCCIONES DE USO
# ============================================================================
print("\n[3/3] Guía de uso del modelo")
print("\n" + "="*80)
print("📖 GUÍA DE USO")
print("="*80)

print("""
🔹 FUNCIONES DISPONIBLES:

1. predecir_flujo(estacion, hora, dia_semana, es_fin_semana)
   Hace una predicción individual
   
   Ejemplo:
   >>> resultado = predecir_flujo('BAQUEDANO', hora=8, dia_semana=0, es_fin_semana=False)
   >>> print(f"Flujo: {resultado['flujo_predicho']:.0f} pasajeros")

2. mostrar_estaciones()
   Muestra todas las estaciones disponibles
   
   Ejemplo:
   >>> mostrar_estaciones()

3. buscar_estacion(palabra)
   Busca estaciones por nombre
   
   Ejemplo:
   >>> buscar_estacion('TOBALABA')

4. comparar_periodos(estacion, dia_semana)
   Compara flujo en diferentes horas del mismo día
   
   Ejemplo:
   >>> comparar_periodos('BAQUEDANO', dia_semana=0)

5. comparar_dias(estacion, hora)
   Compara flujo en diferentes días de la semana
   
   Ejemplo:
   >>> comparar_dias('BAQUEDANO', hora=8)


🔹 PARÁMETROS:

- estacion_nombre: Nombre exacto de la estación (string)
- hora: 0 a 23 (formato 24 horas)
- dia_semana: 0=Lunes, 1=Martes, 2=Miércoles, 3=Jueves, 
              4=Viernes, 5=Sábado, 6=Domingo
- es_fin_semana: True para sábado/domingo, False para días laborales


🔹 PERIODOS AUTOMÁTICOS:

- Punta Mañana: 7:00 - 9:59
- Mediodía: 12:00 - 14:59
- Punta Tarde: 18:00 - 20:59
- Valle: Resto de horas
""")

print("="*80)
print("✅ MODELO LISTO PARA USAR")
print("="*80)

# ============================================================================
# 6. MODO INTERACTIVO (OPCIONAL)
# ============================================================================
print("\n🎮 ¿Quieres hacer una prueba interactiva? (s/n): ", end="")
respuesta = input().strip().lower()

if respuesta == 's':
    print("\n" + "="*80)
    print("MODO INTERACTIVO")
    print("="*80)
    
    # Buscar estación
    print("\n¿Qué estación quieres consultar?")
    print("(Escribe parte del nombre para buscar)")
    busqueda = input(">>> ").strip()
    buscar_estacion(busqueda)
    
    print("\nEscribe el nombre EXACTO de la estación:")
    estacion = input(">>> ").strip().upper()
    
    if estacion in estacion_map:
        print("\n¿Qué hora? (0-23):")
        hora = int(input(">>> ").strip())
        
        print("\n¿Qué día? (0=Lun, 1=Mar, 2=Mié, 3=Jue, 4=Vie, 5=Sáb, 6=Dom):")
        dia = int(input(">>> ").strip())
        
        es_finde = dia >= 5
        
        resultado = predecir_flujo(estacion, hora, dia, es_finde)
        
        if resultado:
            print("\n" + "="*80)
            print("✅ PREDICCIÓN EXITOSA")
            print("="*80)
            print(f"   🚇 Estación: {resultado['estacion']}")
            print(f"   📅 Día: {resultado['dia']}")
            print(f"   🕐 Hora: {resultado['hora']}:00")
            print(f"   ⏰ Periodo: {resultado['periodo']}")
            print(f"\n   👥 FLUJO PREDICHO: {resultado['flujo_predicho']:.0f} pasajeros")
            print("="*80)
    else:
        print(f"❌ Estación '{estacion}' no encontrada")

print("\n✅ Programa finalizado")
print("   Para más predicciones, ejecuta este script nuevamente\n")