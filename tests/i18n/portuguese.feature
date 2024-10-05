# language: pt

Funcionalidade: Somar dois números
  Esquema do Cenário: Somar dois números positivos
    Dado que eu introduzo <Número 1>
    E eu introduzo <Número 2>
    Quando eu somo os números
    Então o resultado deve ser <Resultado>

    Exemplos:
      | Número 1 | Número 2 | Resultado |
      | 2        | 3        | 5         |
      | 4        | 6        | 10        |
