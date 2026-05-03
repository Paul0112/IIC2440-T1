import pandas as pd
import numpy as np
import unicodedata
import pyarrow
import re
import os



"""
LIMPIEZA Y NORMALIZACIÓN
"""
def limpiar_dataset(df):
    """
    Limpieza de datos nulos y cambio
    de fechas a formato estándar
    """
    # Habían id's repetidos en bdd original (se suponía que no deberían haber)
    df.drop_duplicates(subset=['article_id'], keep='last', inplace=True)

    # Cambiar data nula (no eliminar por recuento final)
    df['title'] = df['title'].fillna("sin titulo")
    df['body'] = df['body'].fillna("sin cuerpo")
    df['source'] = df['source'].fillna("fuente desconocida")
    
    # Formato fechas
    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
    df_limpio = df.dropna(subset=['publish_date'])
    
    return df_limpio


def normalizar_texto(text):
    """
    Normaliza una str: 
    minúsculas, quita acentos y caracteres especiales.
    (insensible a mayúsculas y acentos).
    """
    text = str(text) if text is not None else "" # errores numéricos
    s= text.lower()
    n= unicodedata.normalize('NFKD', s)  
    res = ''.join([c for c in n if not unicodedata.combining(c)])  
    
    return res



"""
Construcción de dimensiones y tabla de hechos
"""
def build_date_dim(df):
    """
    Crea la dimensión de tiempo a partir de las fechas del dataset
    """
    # rango max de fechas del df 
    min_date = df['publish_date'].min()
    max_date = df['publish_date'].max()
    
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    dim_date = pd.DataFrame({'full_date': date_range})
    
    # adición de columnas a la dimensión
    dim_date['date_id'] = dim_date['full_date'].dt.strftime('%Y%m%d').astype(int) # LLAVE SIN DUPLICADOS 
    dim_date['year'] = dim_date['full_date'].dt.year
    #dim_date['month'] = dim_date['full_date'].dt.month
    dim_date['month'] = dim_date['full_date'].dt.month.map('{:02d}'.format) # formato de almacenamiento particionado (2 -> 02)

    dim_date['day'] = dim_date['full_date'].dt.day
    dim_date['day_of_week'] = dim_date['full_date'].dt.day_name() 
    
    return dim_date


def build_source_dim(df):
    """
    Crea la dimensión de fuentes de noticias a partir de los datos limpios
    """
    fuentes_unicas = df['source'].unique()
    dim_source = pd.DataFrame({'source_name': fuentes_unicas})
    dim_source['source_id'] = range(1, len(dim_source) + 1) # sk unica sin repetir

    return dim_source


def dicc_regiones():
    """
    Función que define la dim_regiones
    con cada región asociada a un id (retorna dim date).
    Además, se define el diccionario
    de alias para cada una para posterior
    uso en el procesamiento del pipeline (retorna dicc de alias)
    """

    regiones = [
        "Arica y Parinacota",
        "Tarapaca",
        "Antofagasta",
        "Atacama",
        "Coquimbo",
        "Valparaiso",
        "Metropolitana",
        "Ohiggins",
        "Maule",
        "Nuble",
        "Biobio",
        "La Araucania",
        "Los Rios",
        "Los Lagos",
        "Aysen",
        "Magallanes",
        "Desconocida"
    ]

    # construir dimensión región
    dim_region = pd.DataFrame({
        "region_key": range(1, len(regiones) + 1),
        "region_name": regiones
    })

    # Alias 
    region_aliases = {
        1: ["arica", "parinacota"],
        2: ["tarapaca", "iquique", "alto hospicio"],
        3: ["antofagasta", "calama", "tocopilla"],
        4: ["atacama", "copiapo", "vallenar"],
        5: ["coquimbo", "la serena", "ovalle"],
        6: ["valparaiso", "vina del mar", "quilpue", "san antonio"],
        7: ["metropolitana", "santiago", "rm", "region metropolitana", "capital"],
        8: ["ohiggins", "rancagua", "san fernando"],
        9: ["maule", "talca", "curico", "linares"],
        10: ["nuble", "chillan"],
        11: ["biobio", "concepcion", "talcahuano", "los angeles"],
        12: ["araucania", "temuco"],
        13: ["los rios", "valdivia"],
        14: ["los lagos", "puerto montt", "osorno", "castro"],
        15: ["aysen", "coihaique", "coyhaique"],
        16: ["magallanes", "punta arenas"],
        17: []  # desconocida
    }

    return dim_region, region_aliases
    

def construir_patrones(region_aliases):
    """
    Compila la regex una sola vez antes de 
    entrar en el cálculo de cada fila (optimizar)
    """
    patrones = {}

    for key, aliases in region_aliases.items():
        if key == 17:
            continue
        pattern = r'\b(?:' + '|'.join(map(re.escape, aliases)) + r')\b'
        patrones[key] = re.compile(pattern)

    return patrones



def regex_region(df, patrones_listos):
    """
    Para poder inferir la región de la noticia se implementa 
    un sistema de puntuación donde la aparición de la región
    en el titulo vale 3 veces más que en el cuerpo (ya que 
    tiene mas incidencia en el contenido de la noticia).

    Luego la región resultante se elige por votación (es decir,
    la región con mas coincidencias en los aliases gana). En caso
    de empates, se decide por la que tuvo más puntaje en el titulo.
    En caso de un nuevo en este apartado, se escoge la región
    con menor id por construcción.

    Esta versión optimizada (en base a la comentada abajo) crea una
    columna por cada score (por región). Luego, la lógica anterior
    se garantiza al aplicar max sobre cada fila (noticia) devolviendo el id
    asociado a la región que tiene mayor score (aparición ponderada)
    """
    # Trabajamos sobre una copia temporal de columnas para no afectar el df original si algo falla
    score_cols = []

    for key, pattern in patrones_listos.items():
        col_name = f'score_{key}' #formación de columnas scores por región
        
        score_titulo = df['title_clean'].str.count(pattern.pattern) * 3
        score_cuerpo = df['body_clean'].str.count(pattern.pattern)
        
        df[col_name] = (score_titulo + score_cuerpo) 
        score_cols.append(col_name)

    # Encontrar el puntaje máximo por fila
    max_scores = df[score_cols].max(axis=1)
    best_regions = df[score_cols].idxmax(axis=1).str.replace('score_', '').astype(int)
    df['region_id'] = np.where(max_scores < 1, 17, best_regions)
    df = df.drop(columns=score_cols)
    
    return df



"""
VERSION BASE (NO OPTIMIZADA)

def regex_region(titulo, cuerpo, region_aliases):
    '''
    Para poder inferir la región de la noticia se implementa 
    un sistema de puntuación donde la aparición de la región
    en el titulo vale 3 veces más que en el cuerpo (ya que 
    tiene mas incidencia en el contenido de la noticia).

    Luego la región resultante se elige por votación (es decir,
    la región con mas coincidencias en los aliases gana). En caso
    de empates, se decide por la que tuvo más puntaje en el titulo.
    En caso de un nuevo empate en este apartado, se escoge la región
    con menor id.
    '''

    puntajes = {key: 0 for key in region_aliases.keys() if key != 17}

    for key, lista_alias in region_aliases.items():
        if key == 17: continue

        pattern = re.compile(r'\b(' + '|'.join(lista_alias) + r')\b') # Regex SOLO para palabras completas

        # asignación puntajes
        matches_titulo = len(pattern.findall(titulo))
        puntajes[key] += matches_titulo * 3
        matches_cuerpo = len(pattern.findall(cuerpo))
        puntajes[key] += matches_cuerpo * 1

    max_puntos = max(puntajes.values())

    if max_puntos > 0:
        return max(puntajes, key=puntajes.get)
    else:
        return 17 

"""


def build_fact_table(df, dim_source, dim_region, dim_date):
    """
    Ensambla la Fact Table final uniendo los datos de las dimensiones
    creadas anteriormente; además de agregan recuentos en las columnas
    de body y title
    """
    # Recuentos de palabras en cols pedidas
    df['word_count_title'] = df['title_clean'].str.count(r'\s+') + 1
    df['word_count_body'] = df['body_clean'].str.count(r'\s+') + 1
    df.drop(columns=['title_clean', 'body_clean'], inplace=True)

    # JOIN DF CON DIM SOURCE 
    dim_source_idx = dim_source.set_index('source_name') #
    df = df.merge(
        dim_source_idx[['source_id']], 
        left_on='source', # key df
        right_index=True,  
        how='left'
    )
    df.drop(columns=['source'], inplace=True) 

    # JOIN DF DIM DATE 
    dim_date_idx = dim_date.set_index('full_date') 
    df = df.merge(
        dim_date_idx[['date_id', 'year', 'month']], 
        left_on='publish_date', # key df
        right_index=True,  
        how='left'
    )
    df.drop(columns=['publish_date'], inplace=True)


    # JOIN CON  dim region es implicito (ya esta la key cuando se hace la regex)
    df.rename(columns={'region_key': 'region_id'}, inplace=True)
    fact_table = df[['article_id', 'title', 'body', 'word_count_title', 'word_count_body',
                     'date_id', 'source_id', 'region_id', 'year', 'month']].copy()
    
    return fact_table

"""
VALIDACIONES
"""

def validar_consistencia_referencial(fact_news, dim_source, dim_date, dim_region):

    """
    Consistencia referencial: no deben existir llaves foraneas
    (o llaves subrogadas de dimensiones) huerfanas
    """
    source_ok = fact_news['source_id'].isin(dim_source['source_id']).all()
    region_ok = fact_news['region_id'].isin(dim_region['region_key']).all()
    date_ok = fact_news['date_id'].isin(dim_date['date_id']).all()
    
    return source_ok and region_ok and date_ok


def validar_conteo_filas(df_limpio, fact_news):
    """
    El conteo de filas en la tabla de hechos debe
    coincidir con el número de artículos crudos
    """
    return len(fact_news) == len(df_limpio)


def validar_unicidad_dimensiones(dim_source, dim_date, dim_region):
    """
    No deben existir llaves duplicadas dentro
    de una misma tabla de dimensiones
    """
    source_unique = dim_source['source_id'].is_unique
    date_unique = dim_date['date_id'].is_unique
    region_unique = dim_region['region_key'].is_unique
    
    return source_unique and date_unique and region_unique


def validar_particiones(fact_news):
    """ 
    La distribución de particiones debe ser correcta 
    (cada artículo en su partición correspondiente)
    """
    year_ok = fact_news['date_id'] // 10000 == fact_news['year']
    month_ok = (fact_news['date_id'] // 100 % 100) == fact_news['month'].astype(int) #month es str, esntonces 2 -> 02

    return (year_ok & month_ok).all()

def validar_integridad_hechos(fact_news):
    """
    integridad de los datos en la tabla de hechos:
    - asegura que los conteos de palabras sean lógicos (mayores a 0).
    -Asegura que no se hayan duplicado noticias en la tabla de hechos.
    """
    wc_title_ok = (fact_news['word_count_title'] > 0).all()
    wc_body_ok = (fact_news['word_count_body'] > 0).all()
    ids_unicos = fact_news['article_id'].is_unique #unicidad
    
    return wc_title_ok and wc_body_ok and ids_unicos

def run_validation(df_limpio, fact_news, dim_source, dim_date, dim_region):
    """
    Ejecutar todas las funciones de validaciones y comprobar
    si el proceso las pasa todas
    """

    print("\n=== VALIDACIONES ===")

    all_ok = True

    if not validar_consistencia_referencial(fact_news, dim_source, dim_date, dim_region):
        print("Consistencia referencial falló")
        all_ok = False
    else:
        print("Consistencia referencial OK")

    if not validar_conteo_filas(df_limpio, fact_news):
        print("Conteo de filas falló")
        all_ok = False
    else:
        print("Conteo de filas OK")

    if not validar_unicidad_dimensiones(dim_source, dim_date, dim_region):
        print("Unicidad dimensiones falló")
        all_ok = False
    else:
        print("Unicidad dimensiones OK")

    if not validar_particiones(fact_news):
        print("Particiones falló")
        all_ok = False
    else:
        print("Particiones OK")

    if not validar_integridad_hechos(fact_news):
        print("Calidad de hechos falló (Id's duplicados o conteos nulos)")
        all_ok = False
    else:
        print("Calidad de hechos OK")

    return all_ok


"""
EJECUCIÓN
"""
def procesar_etl(input_file):
    '''
    Ejecuta el pipeline construido, aplicando las funciones
    definidas anteriormente.
    '''

    print("Inicio...")
    # Cargar el dataset
    df = pd.read_csv(input_file)
    print("Base de datos leída...")
    
    # Limpiar y normalizar dataset
    df = limpiar_dataset(df)
    df['title_clean'] = df['title'].apply(normalizar_texto)
    df['body_clean'] = df['body'].apply(normalizar_texto)
    print("DataFrame limpio y normalizado...")
        
    # Construcción de dimensiones y fact table
    dim_date = build_date_dim(df)
    dim_source = build_source_dim(df)
    dim_region, region_aliases = dicc_regiones()
    print("Dimensiones creadas...")
    
    patrones_listos = construir_patrones(region_aliases)
    df = regex_region(df, patrones_listos)
    print("Regex correctamente aplicada...")
    #df['region_key'] = df.apply(lambda x: regex_region(x['title_clean'], x['body_clean'], region_aliases), axis=1) # Aplicación de la regex para obtener la region de la noticia

    fact_news = build_fact_table(df, dim_source, dim_region, dim_date)
    print("Tabla de hechos creada...")

    # Guardar dimensiones y almacenamiento particionado
    os.makedirs("warehouse/dim_source", exist_ok=True)
    os.makedirs("warehouse/dim_region", exist_ok=True)
    os.makedirs("warehouse/dim_date", exist_ok=True)
    dim_date.to_parquet("warehouse/dim_date/dim_date.parquet", index=False)
    dim_source.to_parquet("warehouse/dim_source/dim_source.parquet", index=False)
    dim_region.to_parquet("warehouse/dim_region/dim_region.parquet", index=False) 
    print("Almacenamiento de dimensiones completado...")

    fact_news.to_parquet(
        "warehouse/fact_news", 
        partition_cols=["year", "month"], 
        engine="pyarrow", 
        index=False
    )
    print("Almacenamiento particionado completado...")


    # Validaciones
    valido = run_validation(df, fact_news, dim_source, dim_date, dim_region)

    if valido:
        print("ARCHIVO PROCESADO CORRECTAMENTE")
    else:
        print("Error en el procesamiento...")

 

# Ejecución
if __name__ == "__main__":
    procesar_etl('noticias_chile_2023_2025_unique_article_id.csv')