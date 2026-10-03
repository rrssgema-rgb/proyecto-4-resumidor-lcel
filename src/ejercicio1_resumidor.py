from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

#Añadimos el prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Eres un asistente especializado en resumir textos.
Devuelve exactamente 3 viñetas.
Cada viñeta debe contener una idea principal del texto.
No añadas introducción, conclusión ni texto adicional.
Devuelve únicamente las 3 viñetas."""
    ),
    (
        "human",
        "{texto}"
    ),
])

#Añadimos el modelo
modelo = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

#Añadimos el parser
parser = StrOutputParser()


#creamos la cadena LCEL
cadena = prompt | modelo | parser


# PRUEBA 1 Inteligencia artificial
if __name__ == "__main__":

    texto1 = """
    La inteligencia artificial está transformando numerosos sectores de la economía.
    En el sector financiero se utiliza para detectar operaciones fraudulentas,
    analizar grandes cantidades de datos y mejorar la atención al cliente.
    También permite automatizar tareas repetitivas y desarrollar modelos
    predictivos que ayudan a las empresas a tomar decisiones basadas en datos.
    Sin embargo, su utilización requiere controles adecuados para garantizar
    la seguridad, la privacidad y un uso responsable de la información.
    """
   # EJECUTAMOS LA CADENA
   # 1.PREPARAMOS LA ENTRADA
    entrada = {"texto": texto1}
   # 2.EJECUTAMOS LA CADENA Y PREPARAMOS LA SALIDA
    salida = cadena.invoke(entrada)

   # RESULTADOS
    print("\n\n== Texto 1 Inteligencia artificial ==")
   # print("ENTRADA:", entrada["texto"])
   # print("\nSALIDA:")
    print(salida)
    print("\nES TEXTO:", isinstance(salida, str))

# PRUEBA 2 Seguros de automóvil

    texto2 = """
    En los seguros de automóvil, cuando se produce un accidente,
    la compañía debe analizar las circunstancias del siniestro para
    determinar si está cubierto por la póliza. Para ello se revisan
    las garantías contratadas, los daños producidos, la documentación
    aportada y las posibles responsabilidades de las partes implicadas.
    En función de la cobertura, la aseguradora puede asumir los costes
    de reparación del vehículo, los daños a terceros o determinadas
    prestaciones adicionales, como asistencia en carretera o defensa jurídica.
    """

    # 1.PREPARAMOS LA ENTRADA
    entrada2 = {"texto": texto2}

    # 2.EJECUTAMOS LA CADENA
    salida2 = cadena.invoke(entrada2)

    # RESULTADOS
    print("\n\n== Texto 2 Seguros de automóvil ==")
    #print("ENTRADA:", entrada2["texto"])
    #print("\nSALIDA:")
    print(salida2)
    print("\nES TEXTO:", isinstance(salida2, str))    