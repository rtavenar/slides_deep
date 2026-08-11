#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Fonctions de perte, mise à l'échelle, régularisation],
  config-info(
    title: [Deep Learning],
    subtitle: [4. Fonctions de perte, mise à l'échelle, régularisation],
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

= Fonctions de perte

== Fonctions de perte

- Pour entraîner un modèle, il faut *mesurer à quel point il se trompe* : c'est le rôle de la fonction de perte $cal(L)$

- Pertes usuelles
  - *Erreur quadratique moyenne (MSE)* pour la régression \
    #h(1fr) (pénalise les grandes erreurs)
    $ cal(L)(x_i, y_i; theta) = (m(x_i; theta) - y_i)^2 $
  - *Entropie croisée* pour la classification \
    #h(1fr) (pénalise les prédictions erronées mais confiantes)
    $ cal(L)(x_i, y_i; theta) = -log P_theta (y = y_i | x_i) $
    // - binary case ($hat(p) = P_theta(y=1|x_i)$) = logistic loss:
    // $ cal(L) = -[y_i log hat(p) + (1 - y_i) log(1 - hat(p))] $

- Exigence technique pour la descente de gradient : \
  *$cal(L)$ doit être dérivable*

= Mise à l'échelle

== Prétraitement des données

#grid(
  columns: (1.1fr, 1fr),
  gutter: 1em,
  [
    - En pratique, *mettre à l'échelle les caractéristiques d'entrée* est crucial pour un entraînement stable
      // - Without it: slow or unstable convergence
      - Centrer-réduire chaque caractéristique
        $
          tilde(x)_j = (x_j - mu_j) / sigma_j
        $
     - ou mise à l'échelle spécifique au domaine \
        (par ex. images dans [0, 1])
  ], [
    #image-with-caption(image("fig/standardization_fr.svg", width: 100%), [])
  ]
)


== Initialisation des poids

- *Intuition :* garder l'"intensité" du signal (la variance) *constante* à travers les couches
  - trop grande → les activations explosent
  - trop petite → elles s'annulent (disparition des gradients, _cf._ cours suivant)

- *D'où viennent ces formules :* un neurone somme $n_"in"$ entrées → la variance est multipliée par $n_"in"$ → pour la garder stable :

  $ "Var"(w) = 1 / n_"in" $

#pagebreak()

Deux méthodes classiques = raffinements de $"Var"(w) = 1 slash n_"in"$ :

- *Xavier / Glorot* (tanh, sigmoïde) :
  $ w tilde cal(U)(-sqrt(6/(n_"in"+n_"out")), sqrt(6/(n_"in"+n_"out"))) $
  - stable en propagation avant ($n_"in"$) *et* arrière ($n_"out"$) //→ moyenne harmonique
    // - $cal(U)(-a,a)$ has variance $a^2 slash 3$, so $a = sqrt(3 dot 2/(n_"in"+n_"out")) = sqrt(6/(n_"in"+n_"out"))$
  - suppose $phi$ ~linéaire près de 0

- *He* (ReLU) :
  $ w tilde cal(N)(0, sqrt(2/n_"in")) $
  - ReLU supprime ~*la moitié* du signal → x2 sur la variance

// - `keras`: `kernel_initializer` (default `glorot_uniform` for Dense)

// ```python
// Dense(units=256, activation="relu", kernel_initializer="he_normal")
// ```


== Batch Normalization

- *L'initialisation ne fixe l'échelle qu'à $t=0$*
- *Batch Normalization :* même objectif que l'initialisation, mais *imposé pendant l'entraînement* : à *chaque* _mini-batch_
  $ hat(z)^((l)) = (z^((l)) - mu_cal(B)) / sqrt(sigma_cal(B)^2 + epsilon) $
  - puis une échelle $gamma$ et un décalage $beta$ apprenables
- *Mode entraînement vs évaluation :*
  - *Entraînement :* $mu_cal(B), sigma_cal(B)^2$ = statistiques du _mini-batch_ courant
  - *Évaluation :* les statistiques de _mini-batch_ ne sont pas fiables (taille de batch 1, pas de batch) \
    → moyenne glissante $mu, sigma^2$ issue de l'entraînement, figée au moment du test

= Disparition des gradients

== Réseaux de neurones et rétropropagation

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #align(center)[#image("fig/mlp_2hidden_fr.svg", width: 100%)]
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

== Réseaux plus profonds et disparition des gradients


- Des réseaux plus profonds = une compréhension de plus haut niveau
- Mais un problème apparaît : la *disparition des gradients*

- *Intuition :*
  - le gradient remonte à travers *chaque* couche suivante
  - un facteur multiplicatif par couche
  - beaucoup de facteurs $< 1$ \
    → le produit tend vers 0 \
    → les premières couches apprennent à peine
  - *ReLU aide :* $phi' = 1$ (côté actif) → facteur 1, pas d'annulation

= Régularisation

== Sur-paramétrisation en deep learning

- Optimisation (SGD) pour minimiser une fonction de perte
  - Des réseaux plus grands et plus profonds améliorent les performances (à l'entraînement)
  - Risque de surapprentissage

  $ arg min_theta sum_((x_i, y_i) in cal(D)_t) cal(L)(x_i, y_i; theta) != arg min_theta EE_(x,y tilde cal(D)) cal(L)(x, y; theta) $

- Astuces de régularisation
  // - L2 penalty on weights (cf. Ridge regression)
  - Early Stopping (cf. gradient boosting)
  - Dropout (à rapprocher des forêts aléatoires)

== Early Stopping

#image-with-caption(image("fig/early_stopping_fr.svg", width: 65%), [_S'arrêter là où l'erreur de validation est minimale_])

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
    - À chaque _mini-batch_, on *désactive* aléatoirement une fraction des neurones
    - Tous les neurones finissent par être entraînés au cours du processus complet
    - Dans le même esprit que les forêts aléatoires (sous-échantillonnage des caractéristiques)

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
    #image-with-caption(image("fig/srivastava14a.svg", width: 100%), [_Illustration du Dropout — à gauche : réseau complet ; à droite : même réseau avec 40% des neurones désactivés pour un mini-batch. \
    Source : [Srivastava et al., 2014]_])
  ]
)

== Conclusion

- Fonctions de perte
  - Erreur quadratique moyenne pour la régression
  - Perte logistique pour la classification
- Garder des activations bien mises à l'échelle
  - standardisation de l'entrée
  - initialisation → échelle au démarrage
  - BatchNorm → échelle pendant l'entraînement
- Disparition des gradients
  - réseaux plus profonds → le produit tend vers 0
  - ReLU aide (facteur = 1 côté actif)
- Régularisation
  - Early Stopping
  - Dropout
