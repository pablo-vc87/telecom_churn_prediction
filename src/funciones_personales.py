import re

def to_snake_case(col):
    # Buscapalabra normal seguida de mayúsculas (customerID e inserta '_' entre ambos
    col = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', col)
    # munúsculas a las que les sigan mayúscvulas e inserta '_' entre ambos (TotalCharges)
    col = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', col)
    # cambia todo a minúsculas
    return col.lower()