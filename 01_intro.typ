#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Introduction],
  config-info(
    title: [Deep Learning],
    subtitle: [1. Introduction],
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

== Course details

- 24 hours
- Instructor: Romain Tavenard #link("mailto:romain.tavenard@univ-rennes2.fr")[romain.tavenard\@univ-rennes2.fr]
- Tools:
  - Datalab (JupyterHub) at https://datalab.univ-rennes.fr
  - or Jupyter Notebooks running on your machine \
    (need `torch` installed)
- Evaluation (equal weights)
  - Midterm paper exam
  - Research paper reading
  - Lab session

== Pre-requisites

- Basics of Python coding
- A (tiny) bit of calculus
  - What's the derivative of a function?
  - Functions of several variables
- Machine Learning topics
  - Empirical risk optimization and its limitations
  - Model evaluation & selection (cross-validation)

== Our first model: the Perceptron

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - Input: $x = {x_0, dots, x_D}$
    - Output: $a$
    - Parameters to be optimized: ${w_0, dots, w_D, b}$
    - Activation function \
      (chosen a priori): $phi$

    $
      a = phi lr((sum_j w_j x_j + b))
    $
  ],
  [
    #image-with-caption(align(center)[#scale(x: 200%, y: 200%, reflow: true)[#include "cetz/perceptron.typ"]], [_Figure: Perceptron diagram_])
  ]
)

== Motivating example

#grid(
  columns: (55%, 1fr),
  gutter: 1em,
  [
    - #underline[Dataset:] Boston housing prices
    - #underline[Toy task:] predict housing price (PRICE) based on average number of rooms per dwelling (RM)
    - #underline[Chosen model:] linear regression without intercept
      $ "PREDICTED PRICE" = w_0 times "RM" $
    - #underline[Cost function (also called loss):] Mean Squared Error
      $ 1/n sum_(i=1)^n ("PREDICTED PRICE"_(i) - "PRICE"_(i))^2 $
  ],
  [
    #image-with-caption(image("fig/housing_scatter.svg", width: 100%), [_Boston housing: RM vs PRICE_])
  ]
)

== Motivating example: Optimization

#image-with-caption(image("fig/gd_steps.svg", width: 80%), [_Candidate fits and gradient-descent steps on the loss_])
- #underline[Goal:] Find $w_0$ that minimizes our loss
- Testing many values at random is not tractable
- #underline[Alternative strategy:] Gradient descent
  
  $w^((t+1)) arrow.l w^((t)) - rho nabla_w cal(L)(w^((t)))$

== Course outline

- Model architectures
  - Multi-Layer Perceptrons (for tabular data)
  - Convolutional models (for images)
- Different losses for different learning tasks
- Optimization strategies (variants of gradient descent)
