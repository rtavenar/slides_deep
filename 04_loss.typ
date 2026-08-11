#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Loss functions, scaling, regularization],
  config-info(
    title: [Deep Learning],
    subtitle: [4. Loss functions, scaling, regularization],
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

= Loss functions

== Loss functions

- To train a model, we need to *measure how wrong it is*: that's the loss function $cal(L)$

- Standard losses
  - *Mean Squared Error (MSE)* for regression \
    #h(1fr) (penalizes large errors)
    $ cal(L)(x_i, y_i; theta) = (m(x_i; theta) - y_i)^2 $
  - *Cross-entropy* for classification \
    #h(1fr) (penalizes confident wrong predictions)
    $ cal(L)(x_i, y_i; theta) = -log P_theta (y = y_i | x_i) $
    // - binary case ($hat(p) = P_theta(y=1|x_i)$) = logistic loss:
    // $ cal(L) = -[y_i log hat(p) + (1 - y_i) log(1 - hat(p))] $

- Technical requirement for gradient descent: \
  *$cal(L)$ must be differentiable*

= Scaling

== Data preprocessing

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - In practice, *scaling input features* is critical for stable training
      // - Without it: slow or unstable convergence
      - Recommendation: center-reduce each feature
        $
          tilde(x)_j = (x_j - mu_j) / sigma_j
        $    
     - or use domain-specific scaling \
        (e.g. images in [0, 1])
  ], [
    #image-with-caption(image("fig/standardization.svg", width: 100%), [])
  ]
)


== Weight initialization

- *Intuition:* keep signal "loudness" (variance) *constant* across layers
  - too large → activations explode
  - too small → fade to 0 (vanishing gradients, _cf_ next course)

- *Where numbers come from:* neuron sums $n_"in"$ inputs → variance ×$n_"in"$ → to keep it stable:

  $ "Var"(w) = 1 / n_"in" $

#pagebreak()

Two classic schemes = refinements of $"Var"(w) = 1 slash n_"in"$:

- *Xavier / Glorot* (tanh, sigmoid):
  $ w tilde cal(U)(-sqrt(6/(n_"in"+n_"out")), sqrt(6/(n_"in"+n_"out"))) $
  - stable forward ($n_"in"$) *and* backward ($n_"out"$) → harmonic-mean
    // - $cal(U)(-a,a)$ has variance $a^2 slash 3$, so $a = sqrt(3 dot 2/(n_"in"+n_"out")) = sqrt(6/(n_"in"+n_"out"))$
  - assumes $phi$ ~linear near 0

- *He* (ReLU):
  $ w tilde cal(N)(0, sqrt(2/n_"in")) $
  - ReLU kills ~*half* the signal → *double* the variance (factor 2)

// - `keras`: `kernel_initializer` (default `glorot_uniform` for Dense)

// ```python
// Dense(units=256, activation="relu", kernel_initializer="he_normal")
// ```


== Batch Normalization

- *Init fixes scale only at $t=0$* → drifts as weights update
- *Batch Normalization:* Same goal as init, *enforced during training*: re-standardize layer inputs *every* mini-batch
  $ hat(z)^((l)) = (z^((l)) - mu_cal(B)) / sqrt(sigma_cal(B)^2 + epsilon) $
  - then learnable scale $gamma$ + shift $beta$

- *Train vs eval mode:*
  - *Train:* $mu_cal(B), sigma_cal(B)^2$ = statistics of the *current mini-batch*
  - *Eval:* mini-batch stats unreliable (batch size 1, no batch) \
    → *running average* $mu, sigma^2$ from training, kept *fixed* at test time

= Vanishing gradients

== Neural networks and back-propagation

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #align(center)[#image("fig/mlp_2hidden.svg", width: 100%)]
  ],
  [
    #image-with-caption(align(center)[#scale(x: 200%, y: 200%, reflow: true)[#include "cetz/perceptron.typ"]], [])

    $
      (partial a^((l)))/(partial o^((l))) = phi'(o^((l)))
    $

    $
      (partial o^((l)))/(partial a^((l-1))) = w^((l-1))
    $
  ]
)

#pagebreak()

#let cHid1 = rgb(167,149,196)
#let cHid2 = rgb(61,146,140)
#let cOut = rgb(51,159,52)
#let cf(c, body) = text(fill: c, body)

$
  (partial cal(L))/(partial w^((2))) &= cf(cOut, (partial cal(L))/(partial a^((3)))) cf(cOut, (partial a^((3)))/(partial o^((3)))) cf(cOut, (partial o^((3)))/(partial w^((2)))) \

  (partial cal(L))/(partial w^((1))) &= cf(cOut, (partial cal(L))/(partial a^((3)))) cf(cOut, (partial a^((3)))/(partial o^((3)))) cf(cHid2, (partial o^((3)))/(partial a^((2)))) cf(cHid2, (partial a^((2)))/(partial o^((2)))) cf(cHid2, (partial o^((2)))/(partial w^((1)))) \

  (partial cal(L))/(partial w^((0))) &= cf(cOut, (partial cal(L))/(partial a^((3)))) cf(cOut, underbrace((partial a^((3)))/(partial o^((3))), phi^prime (o^((3))))) cf(cHid2, (partial o^((3)))/(partial a^((2)))) cf(cHid2, underbrace((partial a^((2)))/(partial o^((2))), phi^prime (o^((2))))) cf(cHid1, (partial o^((2)))/(partial a^((1)))) cf(cHid1, underbrace((partial a^((1)))/(partial o^((1))), phi^prime (o^((1))))) cf(cHid1, (partial o^((1)))/(partial w^((0))))
$

== Deeper networks and vanishing gradients


- Deeper networks = higher-level understanding
- But a problem appears: *vanishing gradients*

- *Intuition:*
  - gradient travels back through *every* later layer
  - one multiplicative factor per layer
  - many factors $< 1$ \
    → product shrinks to 0 \
    → early layers barely learn
  - *ReLU helps:* $phi' = 1$ (active side) → factor 1, no shrinking

= Regularization

== Over-parametrization in deep learning

- Optimization (SGD) to minimize a loss function
  - Larger & deeper nets improve (training) performance
  - Risks over-fitting

  $ arg min_theta sum_((x_i, y_i) in cal(D)_t) cal(L)(x_i, y_i; theta) != arg min_theta EE_(x,y tilde cal(D)) cal(L)(x, y; theta) $

- Regularization tricks
  // - L2 penalty on weights (cf. Ridge regression)
  - Early stopping (cf. Gradient boosting)
  - Dropout (relates to Random Forests)

== Early Stopping

#image-with-caption(image("fig/early_stopping.svg", width: 65%), [_Stop where validation error is lowest_])

// == Regularization #linebreak() L2 penalty

// - Add a penalty term proportional to the squared norm of weights:

// $
//   cal(L)_r (cal(D); m_theta) = cal(L)(cal(D); m_theta) + lambda sum_l || theta^((l)) ||_2^2
// $

// - Tends to shrink large parameter values → better generalization
// - Equivalent to Ridge regression for linear models

// - In `keras`:

// ```python
// from keras.regularizers import L2

// model = Sequential([
//     InputLayer(input_shape=(d, )),
//     Dense(units=256, activation="relu", kernel_regularizer=L2(0.01)),
//     Dense(units=3, activation="softmax")
// ])
// ```

== Dropout

#grid(
  columns: (55%, 1fr),
  gutter: 1em,
  [
    - At each mini-batch, randomly *switch off* a fraction of neurons
    - All neurons are eventually trained over the full process
    - Similar in spirit to random forests (feature subsampling)

    // - In `keras`, inserted as a layer:

    // ```python
    // from keras.layers import Dropout

    // model = Sequential([
    //     InputLayer(input_shape=(d, )),
    //     Dropout(rate=0.3),
    //     Dense(units=256, activation="relu"),
    //     Dropout(rate=0.3),
    //     Dense(units=3, activation="softmax")
    // ])
    // ```
  ],
  [
    #image-with-caption(image("fig/srivastava14a.svg", width: 100%), [_Dropout illustration — left: full network; right: same network with 40% of neurons switched off for one mini-batch. \
    Source: [Srivastava et al., 2014]_])
  ]
)

== Conclusion

- Loss functions
  - Mean Squared Error for regression
  - Logistic loss for classification
- Keep activations well-scaled
  - input standardization
  - init → scale at start
  - BatchNorm → scale during training
- Vanishing gradients
  - deeper networks → product shrinks to 0
  - ReLU helps (factor = 1 on active side)
- Regularization
  - Early stopping
  - DropOut
