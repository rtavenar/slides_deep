#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Stochastic Gradient Descent],
  config-info(
    title: [Deep Learning],
    subtitle: [2. Stochastic Gradient Descent],
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

== Optimization in general

- *Goal:* Tune model parameters so as to minimize error

- Measuring error
  - Through a cost function / loss function
  - Typical example: Mean Squared Error in regression settings


== Gradient descent

#image-with-caption(image("fig/gd_steps.svg", width: 80%), [])

1. Pick a (differentiable) loss function to be minimized
  e.g. 

  $cal(L)(w, {x_i, y_i}) &= 1/n sum_(i=1)^n cal(L)_i (w, x_i, y_i) \
  &= 1/n sum_(i=1)^n (phi(w^t x_i) - y_i)^2$

2. Use gradient descent
  
  $w^((t+1)) arrow.l w^((t)) - rho nabla_w cal(L)(w^((t)))$

== Gradient descent in Real Life

#image-with-caption(image("fig/gd_pitfalls.svg", width: 100%), [_Learning-rate pitfalls and local minima_])

== Stochastic Gradient Descent (SGD)

#let sgd-color = rgb(131,109,169)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  align: top,
  [
    #algo-box[Algorithm 1: Gradient Descent][
      For each *epoch*:
      + Compute gradient on the *full dataset*:
        #text(fill: sgd-color)[
          $ G &= nabla_w cal(L)(w^((t))) \
          &= 1/n sum_(i=1)^n nabla_w cal(L)_i (w^((t))) $
        ]
      + Update weights:
        $ w^((t+1)) arrow.l w^((t)) - rho G $
    ]
  ],
  [
    #algo-box[Algorithm 2: Mini-Batch SGD][
      For each *epoch*:
      + #text(fill: sgd-color)[Split shuffled data into mini-batches of size $m$]
      + #text(fill: sgd-color)[For each mini-batch $B$]:
        + Estimate gradient on the *mini-batch*:
          #text(fill: sgd-color)[
            $ G approx 1/m sum_(i in B) nabla_w cal(L)_i (w^((t))) $
          ]
        + Update weights:
          $ w^((t+1)) arrow.l w^((t)) - rho G $
    ]
  ]
)

== Gradient Descent vs Stochastic Gradient Descent

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - SGD Cons
      - Subject to high variance
    - SGD Pros
      - Faster weight update (each sample, or each mini-batch)
      - Escape local minima in non-convex settings
  ],
  [
    #image-with-caption(image("fig/gd_vs_sgd.svg", width: 100%), [_GD: smooth path. SGD: noisy path, same minimum_])
  ]
)

== SGD variants: a focus on Adam (1/2)

- *Idea 1*
  - plain SGD: tiny steps on gentle slopes, huge on steep ones
  - want: every parameter advances at a *comparable pace*

- *How:* divide each gradient by *its own magnitude* → size cancels, only *sign* survives:

$
  bold(w)^((t+1)) arrow.l bold(w)^((t)) - rho / sqrt(nabla_w cal(L) dot.o nabla_w cal(L)) dot.o nabla_w cal(L)
  = bold(w)^((t)) - rho dot.o "sign"(nabla_w cal(L))
$

→ *equal-sized step* in every direction \
  (In practice, a small $epsilon$ is added in $sqrt(dot)$ to avoid /0)

== SGD variants: a focus on Adam (2/2)

- *Idea 2*: add momentum
  - single mini-batch gradient = noisy
  - keep a *running average* → like a ball rolling downhill #link("https://distill.pub/2017/momentum/")[[distill]]

- *How:* smooth the gradient (and its square) into averages $bold(m)$, $bold(s)$:

$
  bold(m)^((t+1)) &= beta_1 bold(m)^((t)) + (1 - beta_1) nabla_w cal(L) \
  bold(s)^((t+1)) &= beta_2 bold(s)^((t)) + (1 - beta_2) nabla_w cal(L) dot.o nabla_w cal(L)
$

// - *Bias correction* (early steps start near 0): $hat(bold(m)) = bold(m)^((t+1)) slash (1 - beta_1^t)$, $hat(bold(s)) = bold(s)^((t+1)) slash (1 - beta_2^t)$

- *Full update* uses smoothed quantities:

$
  bold(w)^((t+1)) arrow.l bold(w)^((t)) - rho hat(bold(m)) slash.o sqrt(hat(bold(s)) + epsilon)
$

== Conclusion

- Stochastic Gradient Descent
  - Gradient estimates computed on mini-batches
  - More frequent updates
  - Adam is an extremely powerful variant
