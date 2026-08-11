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

== Détails du cours

- 24 heures
- Enseignant : Romain Tavenard \
  #link("mailto:romain.tavenard@univ-rennes2.fr")[romain.tavenard\@univ-rennes2.fr]
- Outils :
  - Datalab (JupyterHub) sur https://datalab.univ-rennes.fr
  - ou des notebooks Jupyter exécutés sur votre machine \
    (nécessite `torch` + `torchvision`)
- Évaluation
  - Examen écrit de mi-parcours
  - Lecture d'article de recherche
  - Séance de travaux pratiques

== Prérequis

- Notions de base de programmation Python
- Un (tout petit) peu de calcul différentiel
  - Qu'est-ce que la dérivée d'une fonction ?
  - Fonctions de plusieurs variables
- Notions de Machine Learning
  - Optimisation du risque empirique et ses limites
  - Évaluation et sélection de modèles (validation croisée)

== Notre premier modèle : le Perceptron

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - Entrée : $x = {x_0, dots, x_D}$
    - Sortie : $a$
    - Paramètres à optimiser : ${w_0, dots, w_D, b}$
    - Fonction d'activation \
      (choisie a priori) : $phi$

    $
      a = phi lr((sum_j w_j x_j + b))
    $
  ],
  [
    #image-with-caption(align(center)[#scale(x: 200%, y: 200%, reflow: true)[#include "cetz/perceptron.typ"]], [_Figure : schéma du Perceptron_])
  ]
)

== Exemple introductif

#grid(
  columns: (55%, 1fr),
  gutter: 1em,
  [
    // - #underline[Jeu de données :] prix de logements à Boston
    - #underline[Tâche :] prédire le prix d'un logement (PRIX) à partir du nombre moyen de pièces par logement (RM)
    - #underline[Modèle choisi :] régression linéaire sans ordonnée à l'origine
      $ "PRIX PRÉDIT" = w_0 times "RM" $
    - #underline[Fonction de coût (aussi appelée _loss_) :] erreur quadratique moyenne
      $ 1/n sum_(i=1)^n ("PRIX PRÉDIT"_(i) - "PRIX"_(i))^2 $
  ],
  [
    #image-with-caption(image("fig/housing_scatter_fr.svg", width: 100%), [_Prix des logements à Boston : RM vs PRIX_])
  ]
)

== Exemple introductif : optimisation

#image-with-caption(image("fig/gd_steps_fr.svg", width: 80%), [_Régressions candidates et descente de gradient sur la perte_])
- #underline[Objectif :] trouver $w_0$ qui minimise notre perte
- Tester de nombreuses valeurs au hasard n'est pas envisageable
- #underline[Stratégie alternative :] descente de gradient

  $w^((t+1)) arrow.l w^((t)) - rho nabla_w cal(L)(w^((t)))$

== Plan du cours

- Architectures de modèles
  - Perceptrons multicouches (pour les données tabulaires)
  - Modèles convolutifs (pour les images)
- Différentes pertes pour différentes tâches d'apprentissage
- Stratégies d'optimisation (variantes de la descente de gradient)
