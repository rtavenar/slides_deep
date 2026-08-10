#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Loss functions, scaling],
  config-info(
    title: [Deep Learning],
    subtitle: [4. Loss functions, scaling],
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


// - In `keras`, preprocessing can be done before fitting or via a `Normalization` layer



// == Optimization #linebreak() Gradient descent

// #grid(
//   columns: (1fr, 1fr),
//   gutter: 1em,
//   [
//     1. Pick a (differentiable) loss function to be minimized

//     e.g. $cal(L)(w, {x_i, y_i}) = 1/n sum_(i=1)^n cal(L)_i(w, x_i, y_i)$
//     $= 1/n sum_(i=1)^n (phi(w^t x_i) - y_i)^2$

//     2. Use gradient descent

//     $w^((t+1)) arrow.l w^((t)) - rho nabla_w cal(L)(w^((t)))$
//   ],
//   [
//     #figure-placeholder(100%, 150pt, legend: [_Figure: Loss curve $cal(L)$ vs $w$ with gradient descent steps_])

//     #figure-placeholder(100%, 150pt, legend: [_Figure: Algorithm 1 — Gradient Descent pseudocode_])
//   ]
// )

// == Optimization #linebreak() Gradient descent in Real Life

// #image-with-caption(image("fig/gd_pitfalls.svg", width: 100%), [_Learning-rate pitfalls and local minima_])

// == Optimization #linebreak() Stochastic Gradient Descent

// #grid(
//   columns: (1fr, 1fr),
//   gutter: 1em,
//   [
//     #figure-placeholder(100%, 260pt, legend: [_Algorithm 1: Gradient Descent — loop over all $(x_i, y_i)$ per epoch, compute gradient, update weights_])
//   ],
//   [
//     #figure-placeholder(100%, 260pt, legend: [_Algorithm 2: Mini-Batch Stochastic Gradient Descent — loop over mini-batches of size $m$, compute gradient per batch, update weights_])
//   ]
// )

// == Optimization: #linebreak() Gradient Descent vs Stochastic Gradient Descent

// #grid(
//   columns: (1fr, 1fr),
//   gutter: 1em,
//   [
//     - SGD Cons
//       - Subject to high variance
//     - SGD Pros
//       - Faster weight update (each sample, or each mini-batch)
//       - Escape local minima in non-convex settings
//   ],
//   [
//     #image-with-caption(image("fig/gd_vs_sgd.svg", width: 100%), [_GD: smooth path. SGD: noisy path, same minimum_])
//   ]
// )

// == Optimization #linebreak() SGD variants: a focus on Adam (1/2)

// - *Step 1 — idea* (no momentum yet ≈ RMSProp):
//   - plain SGD: tiny steps on gentle slopes, huge on steep ones
//   - want: every parameter advances at a *comparable pace*

// - *How:* divide each gradient by *its own magnitude* → size cancels, only *sign* survives:

// $
//   bold(w)^((t+1)) arrow.l bold(w)^((t)) - rho / sqrt(nabla_w cal(L) dot.o nabla_w cal(L)) dot.o nabla_w cal(L)
//   = bold(w)^((t)) - rho dot.o "sign"(nabla_w cal(L))
// $

// - → *equal-sized step* in every direction
// - #text(size: 0.85em)[(small $epsilon$ added in $sqrt(dot)$ to avoid /0 → step _≈_ sign)]

// == Optimization #linebreak() SGD variants: a focus on Adam (2/2)

// - *Step 2 — idea* (add momentum):
//   - single mini-batch gradient = noisy
//   - keep a *running average* → like a ball rolling downhill #link("https://distill.pub/2017/momentum/")[[distill]]

// - *How:* smooth the gradient (and its square) into averages $bold(m)$, $bold(s)$:

// $
//   bold(m)^((t+1)) = beta_1 bold(m)^((t)) + (1 - beta_1) nabla_w cal(L)
//   quad arrow.r "smoothed gradient"
// $

// $
//   bold(s)^((t+1)) = beta_2 bold(s)^((t)) + (1 - beta_2) nabla_w cal(L) dot.o nabla_w cal(L)
//   quad arrow.r "smoothed squared gradient"
// $

// - *Bias correction* (early steps start near 0): $hat(bold(m)) = bold(m)^((t+1)) slash (1 - beta_1^t)$, $hat(bold(s)) = bold(s)^((t+1)) slash (1 - beta_2^t)$

// - *Full update* = step 1, with smoothed quantities:

// $
//   bold(w)^((t+1)) arrow.l bold(w)^((t)) - rho hat(bold(m)) slash.o sqrt(hat(bold(s)) + epsilon)
// $

// == Optimizing multi-layer perceptron parameters

// - Who wants to compute gradients by hand for such networks (and deeper ones)?

// #align(center)[#image("fig/mlp_2hidden.svg", height: 150pt)]

// $
//   hat(bold(y)) = phi lr([bold(w)^((2)) phi lr((bold(w)^((1)) phi(bold(w)^((0)) bold(x) + b^((0))) + b^((1)))) + b^((2))])
// $

// - *Automatic differentiation:*
//   - you specify only the *forward pass*
//   - framework (keras/torch) computes *all* gradients via backprop
//   - → training code never spells out a derivative

// == Optimization #linebreak() Neural networks and back-propagation

// #grid(
//   columns: (1fr, 1fr),
//   gutter: 1em,
//   [
//     #align(center)[#image("fig/mlp_2hidden.svg", width: 100%)]

//     $
//       (partial cal(L))/(partial w^((2))) = (partial cal(L))/(partial a^((3))) (partial a^((3)))/(partial o^((3))) (partial o^((3)))/(partial w^((2)))
//     $
//     $
//       (partial cal(L))/(partial w^((1))) = (partial cal(L))/(partial a^((3))) (partial a^((3)))/(partial o^((3))) (partial a^((2)))/(partial o^((2))) (partial o^((2)))/(partial w^((1)))
//     $
//     $
//       (partial cal(L))/(partial w^((0))) = (partial cal(L))/(partial a^((3))) dots.h (partial a^((1)))/(partial o^((1))) (partial o^((1)))/(partial w^((0)))
//     $
//   ],
//   [
//     #figure-placeholder(100%, 120pt, legend: [_Figure: Single perceptron node diagram_])

//     $
//       (partial a^((l)))/(partial o^((l))) = phi'(o^((l)))
//     $

//     $
//       (partial o^((l)))/(partial a^((l-1))) = w^((l-1))
//     $
//   ]
// )

// == Deeper networks and vanishing gradients

// #grid(
//   columns: (1fr, 1fr),
//   gutter: 1em,
//   [
//     - Deeper networks = higher-level understanding
//     - But a problem appears: *vanishing gradients*

//     - *Intuition:*
//       - gradient travels back through *every* later layer
//       - one multiplicative factor per layer
//       - many factors $< 1$ → product shrinks to 0
//       - → early layers barely learn
//   ],
//   [
//     #image-with-caption(image("fig/vanishing_grad.svg", width: 100%), [_|gradient| collapses for sigmoid, stays alive for ReLU_])
//   ]
// )

// #pagebreak()

// #text(weight: "bold")[The math that highlights it]

// - Backprop multiplies one activation-derivative $phi'$ per layer:
// $
//   (partial cal(L))/(partial w^((0))) = (partial cal(L))/(partial a^((3))) underbrace(phi'(o^((3))) dot.c phi'(o^((2))) dot.c phi'(o^((1))), "one " phi' " per layer") dots.h
//   quad "with" quad (partial a^((l)))/(partial o^((l))) = phi'(o^((l)))
// $

// - *Vanishes (sigmoid/tanh):* $phi' <= 0.25$ / $phi' <= 1$ → product $arrow.r 0$
// - *ReLU helps:* $phi' = 1$ (active side) → factor 1, no shrinking

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

// == Over-parametrization in deep learning

// - Optimization (SGD) to minimize a loss function
//   - Larger & deeper nets improve (training) performance
//   - Risks over-fitting

//   $ arg min_theta sum_((x_i, y_i) in cal(D)_t) cal(L)(x_i, y_i; theta) != arg min_theta EE_(x,y tilde cal(D)) cal(L)(x, y; theta) $

// - Regularization tricks
//   - L2 penalty on weights (cf. Ridge regression)
//   - Early stopping (cf. Gradient boosting)
//   - Dropout (relates to Random Forests)

// == Regularization #linebreak() Early Stopping

// #image-with-caption(image("fig/early_stopping.svg", width: 65%), [_Stop where validation error is lowest_])

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

// == Regularization #linebreak() Dropout

// #grid(
//   columns: (1fr, 1fr),
//   gutter: 1em,
//   [
//     - At each mini-batch, randomly *switch off* a fraction of neurons
//     - All neurons are eventually trained over the full process
//     - Similar in spirit to random forests (feature subsampling)

//     - In `keras`, inserted as a layer:

//     ```python
//     from keras.layers import Dropout

//     model = Sequential([
//         InputLayer(input_shape=(d, )),
//         Dropout(rate=0.3),
//         Dense(units=256, activation="relu"),
//         Dropout(rate=0.3),
//         Dense(units=3, activation="softmax")
//     ])
//     ```
//   ],
//   [
//     #figure-placeholder(100%, 240pt, legend: [_Figure: Dropout illustration — left: full network; right: same network with 40% of neurons (grey) switched off for one mini-batch. Source: [Srivastava et al., 2014]_])
//   ]
// )

== Conclusion

- Loss functions
  - Mean Squared Error for regression
  - Logistic loss for classification
- Keep activations well-scaled
  - input standardization
  - init → scale at start
  - BatchNorm → scale during training
