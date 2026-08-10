#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Multi Layer Perceptrons],
  config-info(
    title: [Deep Learning],
    subtitle: [3. Multi Layer Perceptrons],
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

== Limitations of the Perceptron

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - Input: $x = {x_0, dots, x_D}$
    - Output: \
      $a = phi(sum_j w_j x_j + b)$

    - Can do
      - Linear regression
      - Linear boundaries for classification
      - Single output
  ],
  [
    #image-with-caption(align(center)[#scale(x: 200%, y: 200%, reflow: true)[#include "cetz/perceptron.typ"]], [_Figure: Perceptron diagram_])
  ]
)

== Multi-Layer Perceptron (MLP) model #linebreak() (Rumelhart, Hinton & Williams, 1985)

// *Definition*

// A Multilayer perceptron is an acyclic graph of neurons, where neurons are structured in successive layers, beginning by an input layer and finishing with an output layer.

#align(center)[#image("fig/mlp_2hidden_wide.svg", width: 100%)]

#pagebreak()

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #align(center)[#image("fig/mlp_2hidden.svg", width: 100%)]
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

== Why introduce hidden layer(s)? #linebreak() Universal approximation theorem (Cybenko, 1989)

- *Intuition:* hidden layers allow the network to learn non-linear transformations of the input — they reshape the data into a space where the final (linear) output layer can solve the task

- *Theoretical result:* a single hidden layer with enough neurons can approximate *any* reasonable function arbitrarily well

- *Practical takeaway:*
  - one hidden layer is sufficient *in theory*
  - but the required number of neurons may be huge
  - in practice, *stacking more (smaller) layers* is more efficient and generalizes better

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

== Building blocks of neural networks #linebreak() Input/Output layers

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #align(center)[#image("fig/mlp_2hidden.svg", width: 100%)]
  ],
  [
    Given a dataset (X, y)
    - Constraints on model structure:
      - Input layer dimension is the number of features in X
      - Output layer has as many units as columns in y
  ]
)

== Building blocks of neural networks #linebreak() Activation functions

- Important features
  - $phi$ should be differentiable almost everywhere
  - Non-linearities
  - Some linear regime
- Examples

#image-with-caption(image("fig/activations.svg", width: 95%), [])

== Building blocks of neural networks #linebreak() Activation functions: the reign of ReLU

- ReLU has become the default choice *for internal layers* over time
- 2 main reasons:
  - cheap to compute (both ReLU and its derivative)
  - vanishing gradients phenomenon (more on that later)

#image-with-caption(image("fig/activations_relu.svg", width: 95%), [_ReLU is now the default for internal layers_])

== Building blocks of neural networks #linebreak() Activation functions: the case of the output layer

- Output activation functions drive the values:
  - identity ("linear" in keras): any real
  - ReLU: any positive value
  - sigmoid: any value in [0, 1]
  - softmax: >0 and sums to 1 (across output neurons)
    #align(center)[#image("fig/softmax.svg", width: 70%)]

// - *Natural pairing:* softmax/sigmoid output = probability → *cross-entropy* loss (next chapter)
