# metro_predictor_corregido_final.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class MetroPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.estaciones_combinacion = [
            'SAN PABLO', 'LOS HEROES', 'UNIVERSIDAD DE CHILE', 'BAQUEDANO', 'LOS LEONES',
            'TOBALABA','LA CISTERNA', 'FRANKLIN', 'SANTA ANA','PUENTE CAL Y CANTO','ÑUÑOA',
            'IRARRAZAVAL','PLAZA DE ARMAS','PLAZA EGAÑA','VICENTE VALDES','VICUÑA MACKENNA',
            'ÑUBLE'
        ]
        self.estaciones_terminales = [
            'LA CISTERNA', 'VESPUCIO NORTE', 'LOS DOMINICOS', 'SAN PABLO',
            'FERNANDO CASTILLO VELASCO', 'PLAZA EGAÑA', 'LOS LIBERTADORES',
            'PLAZA PUENTA ALTO', 'TOBALABA', 'VICENTE VALDES','PLAZA DE MAIPU',
            'LOS LEONES', 'CERRILLOS','VICUÑA MACKENNA', 'LA CISTERNA'
        ]
        
        # Mapeo de estación a comuna (ejemplos - completar con datos reales)
        self.mapeo_estacion_comuna = {
            'ALCANTARA': 'LAS CONDES',
            'TOBALABA': 'LAS CONDES',
            'LOS LEONES': 'PROVIDENCIA',
            'BAQUEDANO': 'SANTIAGO',
            'SANTA ANA': 'SANTIAGO',
            'LA CISTERNA': 'LA CISTERNA',
            'VICENTE VALDES': 'LA CISTERNA',
            'PUENTE ALTO': 'PUENTE ALTO',
            'LOS HEROES': 'SANTIAGO',
            'ESCUELA MILITAR': 'LAS CONDES',
            'LOS DOMINICOS': 'LAS CONDES',
            'SAN PABLO': 'SAN PABLO',
            'LA MONEDA': 'SANTIAGO',
            'VESPUCIO NORTE': 'HUECHURABA'
            # Agregar más estaciones según tu dataset
        }
        
    def cargar_y_limpiar_datos(self, archivo):
        """Carga y limpia el dataset"""
        print("📊 Cargando dataset...")
        df = pd.read_csv(archivo)
        
        # Filtrar registros válidos
        filas_originales = len(df)
        df = df[df['Subidas_Promedio'] > 0.0].copy()
        filas_filtradas = len(df)
        
        print(f"✅ Dataset cargado: {filas_originales} filas originales")
        print(f"✅ Filas válidas después de filtrar: {filas_filtradas}")
        print(f"❌ Filas eliminadas (Subidas_Promedio = 0): {filas_originales - filas_filtradas}")
        
        return df
    
    def feature_engineering(self, df):
        """Crea nuevas características para mejorar el modelo"""
        print("🔧 Aplicando feature engineering...")
        
        # Convertir hora a numérica
        df['hora_numerica'] = pd.to_datetime(df['Media_hora']).dt.hour + pd.to_datetime(df['Media_hora']).dt.minute / 60
        
        # Identificar horas punta
        df['es_hora_punta_manana'] = ((df['hora_numerica'] >= 7) & (df['hora_numerica'] <= 9.5)).astype(int)
        df['es_hora_punta_tarde'] = ((df['hora_numerica'] >= 17.5) & (df['hora_numerica'] <= 20)).astype(int)
        df['es_hora_punta'] = (df['es_hora_punta_manana'] | df['es_hora_punta_tarde']).astype(int)
        
        # Tipo de estación
        def clasificar_estacion(paradero):
            paradero_upper = paradero.upper()
            if paradero_upper in self.estaciones_combinacion:
                return 'Combinacion'
            elif paradero_upper in self.estaciones_terminales:  # CORRECCIÓN: mismo nombre
                return 'Terminal'
            else:
                return 'Regular'
        
        df['tipo_estacion'] = df['Paradero'].apply(clasificar_estacion)
        df['es_combinacion'] = (df['tipo_estacion'] == 'Combinacion').astype(int)
        df['es_terminal'] = (df['tipo_estacion'] == 'Terminal').astype(int)
        
        # CORRECCIÓN CRÍTICA: Determinar Tipo_dia correctamente
        df['Tipo_dia_corregido'] = df['Dia_Semana'].apply(
            lambda x: 'FESTIVO' if x in ['SABADO', 'DOMINGO'] else 'LABORAL'
        )
        
        # Variables de fin de semana
        df['es_fin_de_semana'] = df['Dia_Semana'].isin(['SABADO', 'DOMINGO']).astype(int)
        
        print("✅ Feature engineering completado")
        return df
    
    def preparar_datos(self, df):
        """Prepara los datos para el modelo"""
        print("⚙️ Preparando datos para el modelo...")
        
        # CORRECCIÓN: Usar Tipo_dia_corregido en lugar de Tipo_dia original
        categorical_cols = ['Dia_Semana', 'Tipo_dia_corregido', 'Paradero', 'tipo_estacion']
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Manejar columna renombrada
            col_original = 'Tipo_dia' if col == 'Tipo_dia_corregido' else col
            df[col_original + '_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col_original] = le
        
        # CORRECCIÓN: Definir características finales (SIN comuna - solo 9 características)
        feature_cols = [
            'Dia_Semana_encoded', 'Tipo_dia_encoded', 'Paradero_encoded', 
            'hora_numerica', 'es_hora_punta', 'es_combinacion', 
            'es_terminal', 'es_fin_de_semana', 'tipo_estacion_encoded'
        ]
        
        X = df[feature_cols]
        y = df['Subidas_Promedio']
        
        print(f"✅ Datos preparados: {X.shape[0]} muestras, {X.shape[1]} características")
        print(f"📋 Características usadas: {feature_cols}")
        return X, y
    
    def entrenar_modelo(self, X, y):
        """Entrena el modelo Random Forest"""
        print("🎯 Entrenando modelo Random Forest...")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Crear y entrenar modelo
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = self.model.predict(X_test)
        
        print("\n" + "="*50)
        print("📈 MÉTRICAS DE PERFORMANCE DEL MODELO")
        print("="*50)
        print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
        print(f"MAE (Error Absoluto Medio): {mean_absolute_error(y_test, y_pred):.2f}")
        print(f"RMSE (Raíz del Error Cuadrático Medio): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        
        # Validación cruzada
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        print(f"Validación Cruzada R² (5-fold): {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Características más importantes:")
        for i, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.model
    
    def guardar_modelo(self, archivo_modelo='modelo_metro.pkl', archivo_encoders='encoders.pkl'):
        """Guarda el modelo y encoders entrenados"""
        joblib.dump(self.model, archivo_modelo)
        joblib.dump(self.label_encoders, archivo_encoders)
        
        # Guardar también el mapeo de estaciones a comunas
        joblib.dump(self.mapeo_estacion_comuna, 'mapeo_estaciones.pkl')
        
        print(f"💾 Modelo guardado como: {archivo_modelo}")
        print(f"💾 Encoders guardados como: {archivo_encoders}")
    
    def cargar_modelo(self, archivo_modelo='modelo_metro.pkl', archivo_encoders='encoders.pkl'):
        """Carga el modelo y encoders guardados"""
        try:
            self.model = joblib.load(archivo_modelo)
            self.label_encoders = joblib.load(archivo_encoders)
            self.mapeo_estacion_comuna = joblib.load('mapeo_estaciones.pkl')
            print("✅ Modelo y encoders cargados exitosamente")
            return True
        except FileNotFoundError:
            print("❌ No se encontraron archivos del modelo. Entrena un nuevo modelo primero.")
            return False
    
    def obtener_comuna_por_estacion(self, estacion):
        """Obtiene la comuna automáticamente basándose en la estación"""
        estacion_upper = estacion.upper()
        
        # Buscar coincidencia exacta
        if estacion_upper in self.mapeo_estacion_comuna:
            return self.mapeo_estacion_comuna[estacion_upper]
        
        # Buscar coincidencia parcial
        for estacion_mapeo, comuna in self.mapeo_estacion_comuna.items():
            if estacion_upper in estacion_mapeo or estacion_mapeo in estacion_upper:
                return comuna
        
        # Si no se encuentra, usar una por defecto
        print(f"⚠️  Estación '{estacion}' no encontrada en el mapeo. Usando comuna por defecto.")
        return 'SANTIAGO'  # Comuna por defecto
    
    def determinar_tipo_dia(self, dia_semana):
        """Determina el tipo de día basándose en el día de la semana"""
        dia_upper = dia_semana.upper()
        return 'FESTIVO' if dia_upper in ['SABADO', 'DOMINGO'] else 'LABORAL'
    
    def predecir_flujo(self, dia_semana, hora, estacion):
        """Realiza una predicción basada en los parámetros ingresados"""
        if self.model is None:
            print("❌ Modelo no disponible. Entrena o carga un modelo primero.")
            return None, None
        
        try:
            # Preparar datos de entrada
            hora_dt = pd.to_datetime(hora)
            hora_numerica = hora_dt.hour + hora_dt.minute / 60
            
            es_hora_punta = 1 if ((hora_numerica >= 7 and hora_numerica <= 9.5) or 
                                 (hora_numerica >= 17.5 and hora_numerica <= 20)) else 0
            
            # Determinar tipo de estación - CORRECCIÓN: usar nombre correcto
            estacion_upper = estacion.upper()
            if estacion_upper in self.estaciones_combinacion:
                tipo_estacion = 'Combinacion'
                es_combinacion = 1
                es_terminal = 0
            elif estacion_upper in self.estaciones_terminales:  # CORRECCIÓN: mismo nombre
                tipo_estacion = 'Terminal'
                es_combinacion = 0
                es_terminal = 1
            else:
                tipo_estacion = 'Regular'
                es_combinacion = 0
                es_terminal = 0
            
            es_fin_de_semana = 1 if dia_semana.upper() in ['SABADO', 'DOMINGO'] else 0
            
            # CORRECCIÓN CRÍTICA: Determinar tipo_dia correctamente
            tipo_dia = self.determinar_tipo_dia(dia_semana)
            
            # Codificar variables categóricas
            dia_encoded = self.label_encoders['Dia_Semana'].transform([dia_semana.upper()])[0]
            tipo_dia_encoded = self.label_encoders['Tipo_dia'].transform([tipo_dia])[0]
            paradero_encoded = self.label_encoders['Paradero'].transform([estacion.upper()])[0]
            tipo_estacion_encoded = self.label_encoders['tipo_estacion'].transform([tipo_estacion])[0]
            
            # CORRECCIÓN CRÍTICA: Crear array con solo 9 características (SIN comuna)
            features = np.array([[
                dia_encoded, tipo_dia_encoded, paradero_encoded,
                hora_numerica, es_hora_punta, es_combinacion, es_terminal,
                es_fin_de_semana, tipo_estacion_encoded
            ]])
            
            # Realizar predicción
            prediccion = self.model.predict(features)[0]
            
            # Obtener comuna para mostrar (solo informativo)
            comuna_detectada = self.obtener_comuna_por_estacion(estacion)
            
            return max(0, round(prediccion, 2)), comuna_detectada
            
        except Exception as e:
            print(f"❌ Error en la predicción: {e}")
            import traceback
            print(f"🔍 Detalles: {traceback.format_exc()}")
            return None, None

def mostrar_menu():
    """Muestra el menú principal"""
    print("\n" + "="*60)
    print("🚇 PREDICTOR DE FLUJO DE PASAJEROS - METRO DE SANTIAGO")
    print("="*60)
    print("1. 🔄 Entrenar nuevo modelo")
    print("2. 📊 Realizar predicción")
    print("3. ℹ️  Mostrar información del modelo")
    print("4. 🚪 Salir")
    print("="*60)

def main():
    predictor = MetroPredictor()
    
    # Intentar cargar modelo existente
    modelo_cargado = predictor.cargar_modelo()
    
    while True:
        mostrar_menu()
        opcion = input("\nSeleccione una opción (1-4): ").strip()
        
        if opcion == '1':
            # Entrenar nuevo modelo
            archivo = input("Ingrese el nombre del archivo CSV (ej: metro_santiago_dataset_limpio.csv): ").strip()
            try:
                df = predictor.cargar_y_limpiar_datos(archivo)
                df = predictor.feature_engineering(df)
                X, y = predictor.preparar_datos(df)
                predictor.entrenar_modelo(X, y)
                predictor.guardar_modelo()
                modelo_cargado = True
                
                print("\n✅ ¡Modelo entrenado exitosamente! Ahora puede usar la opción 2 para predicciones.")
                input("Presione Enter para continuar...")
                
            except Exception as e:
                print(f"❌ Error al entrenar el modelo: {e}")
                input("Presione Enter para continuar...")
        
        elif opcion == '2':
            # Realizar predicción
            if not modelo_cargado:
                print("❌ Primero debe entrenar o cargar un modelo (Opción 1)")
                input("Presione Enter para continuar...")
                continue
            
            print("\n🎯 INGRESE LOS DATOS PARA LA PREDICCIÓN:")
            print("-" * 40)
            
            dia_semana = input("Día de la semana (Lunes, Martes, ..., Domingo): ").strip()
            hora = input("Hora (formato HH:MM, ej: 08:30, 17:45): ").strip()
            estacion = input("Estación (ej: Tobalaba, Los Héroes, La Cisterna): ").strip()
            
            print("\n⏳ Realizando predicción...")
            prediccion, comuna_detectada = predictor.predecir_flujo(dia_semana, hora, estacion)
            
            if prediccion is not None:
                print("\n" + "="*50)
                print("📊 RESULTADO DE LA PREDICCIÓN")
                print("="*50)
                print(f"📍 Estación: {estacion.title()}")
                print(f"🏙️  Comuna detectada: {comuna_detectada.title()}")
                print(f"📅 Día: {dia_semana.title()}")
                print(f"⏰ Hora: {hora}")
                print(f"👥 Flujo estimado: {prediccion:.0f} personas")
                print("="*50)
                
                # Interpretación
                if prediccion < 50:
                    intensidad = "Muy bajo"
                elif prediccion < 100:
                    intensidad = "Bajo"
                elif prediccion < 200:
                    intensidad = "Moderado"
                elif prediccion < 400:
                    intensidad = "Alto"
                else:
                    intensidad = "Muy alto"
                
                print(f"💡 Intensidad de flujo: {intensidad}")
            else:
                print("❌ No se pudo realizar la predicción. Verifique los datos ingresados.")
            
            input("\nPresione Enter para continuar...")
        
        elif opcion == '3':
            # Información del modelo
            if modelo_cargado:
                print("\n📋 INFORMACIÓN DEL MODELO")
                print("-" * 30)
                print("🔧 Algoritmo: Random Forest Regressor")
                print("📊 Características utilizadas: 9")
                print("🎯 Objetivo: Predecir flujo de pasajeros")
                print("💾 Estado: Modelo cargado y listo")
                print(f"📍 Estaciones mapeadas: {len(predictor.mapeo_estacion_comuna)}")
                print(f"🔢 Encoders disponibles: {list(predictor.label_encoders.keys())}")
            else:
                print("❌ No hay modelo cargado. Use la opción 1 para entrenar uno.")
            
            input("\nPresione Enter para continuar...")
        
        elif opcion == '4':
            print("\n👋 ¡Hasta luego! Gracias por usar el predictor de flujo.")
            break
        
        else:
            print("❌ Opción inválida. Por favor, seleccione 1-4.")
            input("Presione Enter para continuar...")

if __name__ == "__main__":
    print("🚇 INICIANDO PREDICTOR DE FLUJO - METRO DE SANTIAGO")
    print("Cargando sistema...")
    main()
