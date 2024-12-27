from fastapi import FastAPI
import efficient_attention
import time
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

# Definir el modelo para la matriz
class Matrix(BaseModel):
    matrix: List[List[float]]

# Definir el modelo para el vector
class VectorS(BaseModel):
    vector: List[float]

class VectorI(BaseModel):
    vector: List[int]

@app.post("/efficient-attention")
async def calculo(dim_heads: int, queries: Matrix,
                  keys: Matrix, values: Matrix):
    start = time.time()

    attention = efficient_attention.EfficientAttention(dim_heads)

    # Calcular atención
    output = attention.forward(queries.matrix, keys.matrix, values.matrix)
    
    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "Atencion": output
    }
    jj = json.dumps(str(j1))

    return jj