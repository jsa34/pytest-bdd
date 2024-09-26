# language: fr

Fonctionnalité: Additionner deux nombres
  Plan du scénario: Additionner deux nombres positifs
    Étant donné que j'introduis <Nombre 1>
    Et j'introduis <Nombre 2>
    Quand j'additionne les nombres
    Alors le résultat doit être <Résultat>

      Exemples:
      | Nombre 1 | Nombre 2 | Résultat |
      | 2        | 3        | 5        |
      | 4        | 6        | 10       |
