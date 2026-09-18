#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Descente de Gradient Stochastique],
  config-info(
    title: [Deep Learning],
    subtitle: [2. Descente de Gradient Stochastique],
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

== Optimisation en général

- *Objectif :* ajuster les paramètres du modèle afin de minimiser l'erreur

- Mesurer l'erreur
  - Via une fonction de coût / fonction de perte
  - Exemple typique : l'erreur quadratique moyenne en régression


== Descente de gradient

#image-with-caption(image("fig/gd_steps_fr.svg", width: 80%), [])

1. Choisir une fonction de perte (différentiable) à minimiser :

  $cal(L)(w, {x_i, y_i}) &= 1/n sum_(i=1)^n cal(L)_i (w, x_i, y_i) \
  &= 1/n sum_(i=1)^n (phi(w^t x_i) - y_i)^2$

2. Appliquer itérativement des pas de descente de gradient

  $w^((t+1)) arrow.l w^((t)) - rho nabla_w cal(L)(w^((t)))$

== La descente de gradient en pratique

#image-with-caption(image("fig/gd_pitfalls_fr.svg", width: 100%), [_Pièges du taux d'apprentissage et minima locaux_])

== Descente de Gradient Stochastique (SGD)

#let sgd-color = rgb(131,109,169)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  align: top,
  [
    #algo-box[Algorithme 1 : Descente de gradient][
      Pour chaque _epoch_ :
      + Calculer le gradient sur le *jeu de données complet* :
        #text(fill: sgd-color)[
          $ G &= nabla_w cal(L)(w^((t))) \
          &= 1/n sum_(i=1)^n nabla_w cal(L)_i (w^((t))) $
        ]
      + Mettre à jour $w$ :
        $ w^((t+1)) arrow.l w^((t)) - rho G $
    ]
  ],
  [
    #algo-box[Algorithme 2 : SGD par _mini-batches_][
      Pour chaque _epoch_ :
      + #text(fill: sgd-color)[Découper les données en _mini-batches_ de taille $m$]
      + #text(fill: sgd-color)[Pour tout _mini-batch_ $B$:]
        + Estimer le gradient sur le *_mini-batch_* :
          #text(fill: sgd-color)[
            $ G approx 1/m sum_(i in B) nabla_w cal(L)_i (w^((t))) $
          ]
        + Mettre à jour $w$ :
          $ w^((t+1)) arrow.l w^((t)) - rho G $
    ]
  ]
)

== Descente de gradient contre descente de gradient stochastique

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - Inconvénients du SGD
      - Sujet à une forte variance
    - Avantages du SGD
      - Mise à jour des poids plus rapide (à chaque exemple, ou à chaque _mini-batch_)
      - Permet d'échapper aux minima locaux dans les cas non convexes
  ],
  [
    #image-with-caption(image("fig/gd_vs_sgd_fr.svg", width: 100%), [_GD : trajectoire lisse. SGD : trajectoire bruitée, même minimum_])
  ]
)

== Variantes du SGD : zoom sur Adam (1/2)

- *Idée 1*
  - SGD classique : petits pas sur les pentes douces, immenses sur les pentes raides
  - objectif : que chaque paramètre progresse à un *rythme comparable*

- *Comment :* diviser chaque gradient par *sa propre magnitude* → la taille s'annule, seul le *signe* subsiste :

$
  bold(w)^((t+1)) arrow.l & bold(w)^((t)) - rho underbrace((nabla_w cal(L)) / sqrt(nabla_w cal(L) dot.o nabla_w cal(L)), "sign"(nabla_w cal(L)))
$

→ un *pas de même taille* dans toutes les directions
  // (en pratique, un petit $epsilon$ est ajouté dans $sqrt(dot)$ pour éviter les divisions par 0)

== Variantes du SGD : zoom sur Adam (2/2)

- *Idée 2* : ajouter du _momentum_
  - le gradient d'un seul _mini-batch_ est bruité
  - conserver une *moyenne glissante* → comme une balle qui dévale une pente #link("https://distill.pub/2017/momentum/")[[distill]]

- *Comment :* lisser le gradient (et son carré) en moyennes $bold(m)$, $bold(s)$ :

$
  bold(m)^((t+1)) &= beta_1 bold(m)^((t)) + (1 - beta_1) nabla_w cal(L) \
  bold(s)^((t+1)) &= beta_2 bold(s)^((t)) + (1 - beta_2) nabla_w cal(L) dot.o nabla_w cal(L)
$

// - *Bias correction* (early steps start near 0): $hat(bold(m)) = bold(m)^((t+1)) slash (1 - beta_1^t)$, $hat(bold(s)) = bold(s)^((t+1)) slash (1 - beta_2^t)$

- *Mise à jour complète* utilisant les quantités lissées :

$
  bold(w)^((t+1)) arrow.l bold(w)^((t)) - rho hat(bold(m)) slash.o sqrt(hat(bold(s)) + epsilon)
$

== Conclusion

- Descente de Gradient Stochastique
  - Estimations du gradient calculées sur des _mini-batches_
  - Mises à jour plus fréquentes
  - Adam est une variante extrêmement performante
