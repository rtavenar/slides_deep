#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Perceptrons Multi-Couches],
  config-info(
    title: [Deep Learning],
    subtitle: [3. Perceptrons Multi-Couches],
    author: [Romain Tavenard],
    date: []
  ),
  config-colors(
    primary: rgb(131,109,169),
    secondary: rgb(200,200,200),
    tertiary: rgb(200,200,200),
    primary-light: rgb(200,200,200)
  )
)

#set text(font: "Helvetica Neue", weight: "light")
#show link: underline

#title-slide()

== Limites du Perceptron

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - Entrée : $x = {x_0, dots, x_D}$
    - Sortie : \
      $a = phi(sum_j w_j x_j + b)$

    - Peut faire
      - Régression linéaire
      - Frontières linéaires pour la classification
      - Sortie unique
  ],
  [
    #image-with-caption(align(center)[#scale(x: 200%, y: 200%, reflow: true)[#include "cetz/perceptron.typ"]], [_Figure : schéma du Perceptron_])
  ]
)

== Modèle du Perceptron Multi-Couches (MLP) #linebreak() (Rumelhart, Hinton & Williams, 1985)

// *Definition*

// A Multilayer perceptron is an acyclic graph of neurons, where neurons are structured in successive layers, beginning by an input layer and finishing with an output layer.

#align(center)[#image("fig/mlp_2hidden_wide_fr.svg", width: 100%)]

#pagebreak()

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #align(center)[#image("fig/mlp_2hidden_fr.svg", width: 100%)]
  ],
  [
    $
      hat(y) = phi_"out" lr((sum_i w_i^((2)) h_i^((2)) + b^((2))))
    $
    $
      forall i, h_i^((2)) = phi lr((sum_j w_(i j)^((1)) h_j^((1)) + b_i^((1))))
    $
    $
      forall i, h_i^((1)) = phi lr((sum_j w_(i j)^((0)) x_j + b_i^((0))))
    $
  ]
)

== Pourquoi introduire des couches cachées ?

- *Intuition :* les couches cachées permettent au réseau d'apprendre des transformations non linéaires de l'entrée — elles remodèlent les données dans un espace où la couche de sortie (linéaire) peut résoudre la tâche

- *Résultat théorique* (théorème d'approximation universelle, Cybenko, 1989) *:* une seule couche cachée, avec suffisamment de neurones, peut approximer *n'importe quelle* fonction raisonnable avec une précision arbitraire

- *À retenir en pratique :*
  - une couche cachée suffit *en théorie*
  - mais le nombre de neurones nécessaires peut être énorme
  - en pratique, *empiler davantage de couches (plus petites)* est plus efficace et généralise mieux

// == Why introduce hidden layer(s)? #linebreak() In practice

// - Universal Approximation Theorem:
//   - A single hidden layer is sufficient in theory
//   - The number of neurons in this layer is not given

// - In practice, for a given approximation level
//   - Stacking more layers requires less parameters
//   - ⚠ Very deep networks suffer from specific problems too (discussed later in the course)
//   - 2-3 hidden layers is a good start for an MLP

// == Multi-Layer Perceptron in `keras`

// ```python
// from keras.models import Sequential
// from keras.layers import InputLayer, Dense

// # Regression MLP: 10 inputs → 2 hidden layers → 1 output
// model = Sequential([
//     InputLayer(input_shape=(10,)),
//     Dense(units=64, activation="relu"),
//     Dense(units=64, activation="relu"),
//     Dense(units=1, activation="linear")  # regression output
// ])

// model.summary()
// ```

// - `activation` sets the activation function per layer
// - Output activation depends on the task (see next slides)

// == End-to-end learning

// - Classification using MLP
//   - Hidden layers: non-linear transformations
//   - Last layer: logistic regression
// - Example

// #image-with-caption(image("fig/spiral.svg", width: 45%), [_Spiral data: hidden layers reshape it so a linear output can separate the classes_])

== Briques de base des réseaux de neurones #linebreak() Couches d'entrée/sortie

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #align(center)[#image("fig/mlp_2hidden_fr.svg", width: 100%)]
  ],
  [
    Étant donné un jeu de données (X, y)
    - Contraintes sur la structure du modèle :
      - La dimension de la couche d'entrée est le nombre de caractéristiques dans X
      - La couche de sortie possède autant d'unités que de colonnes dans y
  ]
)

== Briques de base des réseaux de neurones #linebreak() Fonctions d'activation

- Propriétés importantes
  - $phi$ doit être dérivable presque partout
  - Non-linéarités
  - Un régime quasi linéaire
- Exemples

#image-with-caption(image("fig/activations_fr.svg", width: 95%), [])

== Briques de base des réseaux de neurones #linebreak() Fonctions d'activation : ReLU domine

- ReLU est devenue le choix par défaut *pour les couches internes*
- 2 raisons principales :
  - peu coûteuse à calculer (ReLU et sa dérivée)
  - disparition des gradients (davantage de détails plus tard)

#image-with-caption(image("fig/activations_relu_fr.svg", width: 95%), [_ReLU est aujourd'hui le choix par défaut pour les couches internes_])

== Briques de base des réseaux de neurones #linebreak() Fonctions d'activation : le cas de la couche de sortie

- Les fonctions d'activation de sortie déterminent les valeurs :
  - identité ("linear" dans keras) : tout réel
  - ReLU : toute valeur positive
  - sigmoïde : toute valeur dans [0, 1]
  - softmax : >0 et somme à 1 (sur les neurones de sortie)
    #align(center)[#image("fig/softmax_fr.svg", width: 70%)]

// - *Natural pairing:* softmax/sigmoid output = probability → *cross-entropy* loss (next chapter)
