#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Images et réseaux de neurones convolutifs],
  config-info(
    title: [Deep Learning],
    subtitle: [5. Images et réseaux de neurones convolutifs],
    author: [Romain Tavenard],
    date: [
      #align(bottom)[
        #text(size: 0.8em)[NB : la plupart des figures de ces diapositives proviennent de _Dumoulin & Visin. "A guide to convolution arithmetic for deep learning". 2016_]
      ]
]
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

== Préambule : qu'est-ce qu'une image ?

#image-with-caption(image("fig/rgb_pixels_fr.svg", width: 100%), [])

== L'opérateur de convolution

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - Convolution 2D
      - Bleu : image d'entrée
      - Gris : noyau de convolution
      - Cyan : carte d'activation

    - Opération de convolution = produit scalaire entre
      - le noyau de convolution (aussi appelé filtre)
      - une sous-partie de l'entrée
  ],
  [
    #align(center)[
      #image("fig/conv_no_padding_no_strides.svg", width: 70%) \
      #v(-3em)#image("fig/conv_no_pad.svg", width: 100%)
    ]
  ]
)

== Couches convolutives dans un réseau de neurones (1/2)

- Une couche convolutive est composée de
  - noyaux de convolution
  - biais (1 par noyau)
  - une fonction d'activation

- Utile car
  - réduit le nombre de paramètres (*partage de paramètres* : le même noyau est réutilisé partout)
    - _par ex._ neurone dense sur $224 times 224$ : $approx 50 000$ poids contre $3 times 3$ pour un noyau : 9
  - encode l'*équivariance par translation* (une translation de l'entrée induit une translation de la sortie)

// == Convolution and translation

// #figure-placeholder(100%, 300pt, legend: [Figure: A "4" digit image on left, passed through a conv layer (green), output is a stack of feature maps showing translated activation patterns. Source: Christian Wolf, Twitter https://twitter.com/chriswolfvision/status/1313059518574718977?s=20])

== Couches convolutives dans un réseau de neurones (2/2)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - Cas de plusieurs canaux d'entrée
      - on somme la réponse sur tous les canaux
    - Cas de plusieurs noyaux
      - chaque noyau produit un canal de sortie
  ],
  [
    #image-with-caption(image("fig/conv_multichannel_fr.svg", width: 110%), [])
  ]
)

== Couches convolutives : hyperparamètres

- _Padding_

#grid(
  columns: (1fr, 1fr),
  gutter: 0.5em,
  [
    #v(-1em) #image-with-caption(image("fig/conv_same.svg", width: 80%), [`padding="same"`], caption-align: center)
  ],
  [
    #v(-1em) #image-with-caption(image("fig/conv_valid.svg", width: 80%), [`padding="valid"`], caption-align: center)
  ]
)

- _Strides_

#grid(
  columns: (1fr, 1fr),
  gutter: 0.5em,
  [
    #v(-1em) #image-with-caption(image("fig/conv_stride1.svg", width: 80%), [`stride=1`], caption-align: center)
  ],
  [
    #v(-1em) #image-with-caption(image("fig/conv_stride2.svg", width: 80%), [`stride=2`], caption-align: center)
  ]
)

== Calculer la taille d'une carte d'activation

- Hypothèse : pas de _padding_ ("valid"), _strides_ unitaires

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #image-with-caption(image("fig/conv_valid.svg", width: 100%), [_La sortie rétrécit de (taille du noyau − 1)_])
  ],
  [
    $
      W_"out" = W_"in" - W_k + 1
    $
    $
      H_"out" = H_"in" - H_k + 1
    $

    *Exemple :*
    - Entrée : $28 times 28$ (chiffre MNIST)
    - Noyau : $5 times 5$
    - Sortie : $(28-5+1) times (28-5+1) = 24 times 24$

    *Avec padding="same" :* la sortie reste $28 times 28$
  ]
)

== Couches de _pooling_ (ou sous-échantillonnage)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - *Pourquoi faire du _pooling_ ?*
      - réduit la taille spatiale → moins de calculs
      - robustesse aux petits décalages (conv = équivariante, pooling → invariant)
    - _Max pooling_ / _Average pooling_
    - Hyperparamètre
      - taille du pooling
  ],[
    #image-with-caption(image("fig/pool_max.svg", width: 100%), [_Exemple de max pooling_])
  ]
)

// = A history of Convolutional neural networks (CNN)

== LeNet (1989)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #image-with-caption(image("fig/mnist_grid.svg", width: 100%), [])
  ],
  [
    Jeu de données MNIST \
    10 classes \
    60 000 images
  ]
)

#image-with-caption(image("fig/LeNet5.svg", width: 100%), [_Architecture LeNet-5 (LeCun et al., 1989)_. Source : Wikipedia])

== 2012–2015 : une amélioration radicale des performances

- ImageNet : 15M images, 22k classes
- LSVRC : sous-ensemble d'ImageNet (1,2M images, 1k classes)
- Ingrédients clés :
  - des réseaux plus profonds (⚠ disparition des gradients)
  - régularisation (BatchNorm, augmentation de données...)
#image-with-caption(image("fig/lsvrc_fr.svg", width: 60%), [])

== _ResNet_ (2015) — vraiment très profond

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - *Problème :* au-delà de ~20 couches, plus profond = *pire* (disparition des gradients)
    - *Connexion résiduelle :*
      $ bold(y) = bold(x) + cal(F)(bold(x)) $
    - rien d'utile à ajouter ? on sort $bold(x)$ → la profondeur ne nuit jamais
    - a permis des réseaux de *100+ couches*
  ],
  [
    #image-with-caption(align(center)[#scale(x: 150%, y: 150%, reflow: true)[#include "cetz/resnet_block_fr.typ"]], [_Bloc résiduel : la connexion identité laisse les gradients contourner les couches de poids_])
  ]
)

#pagebreak()

*Pourquoi la connexion résiduelle corrige la disparition des gradients*

- Le gradient qui remonte à travers un bloc résiduel s'écrit :

$
  (partial bold(y))/(partial bold(x)) = underbrace(1, "raccourci identité") + (partial cal(F)(bold(x)))/(partial bold(x))
$

- le *"1" est essentiel* : même si $partial cal(F) slash partial bold(x)$ est minuscule, le gradient reste $approx 1$ (et non $approx 0$)
- empiler $L$ blocs → produit de termes $(1 + dots)$ → *pas d'effondrement vers zéro*

== Batch Normalization pour les images

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - Rappel : BatchNorm normalise les activations (moyenne nulle, variance unitaire), puis les remet à l'échelle avec $gamma, beta$ appris
    - Pour un batch d'images (tenseur $N times C times H times W$)
      - une paire moyenne/variance *par canal* $C$
      // - computed jointly over the batch ($N$) and spatial ($H, W$) dimensions
    - Effet : stabilise/accélère l'entraînement ; agit aussi comme un régularisateur (léger)
  ],
  [
    #image-with-caption(image("fig/batchnorm_cube.svg", width: 100%), [_Batch Norm : pour chaque canal (tranche violette), les statistiques sont calculées sur les dimensions batch et spatiales ($N, H, W$)_])
  ]
)

== Régularisation : augmentation de données

- Rappel : les grands réseaux sur-apprennent → il faut régulariser (early stopping, Dropout, BatchNorm)
- Images : une astuce supplémentaire, très efficace : l'*augmentation de données*
- Principe : générer des exemples d'entraînement virtuels
  // - original image $x_i$
  // - modified image $hat(x)_i$
  // - unchanged label $y_i$

#grid(
  columns: (.8fr, 1fr),
  gutter: 1em,
  align: bottom,
  [
    #image-with-caption(image("fig/aug_original.svg", width: 75%), [_Image originale_ (étiquette : "chat")], image-align: center, caption-align: center)
  ],
  [
    #image-with-caption(image("fig/aug_grid_fr.svg", width: 100%), [_6 versions augmentées (même étiquette)_], caption-align: center)
  ]
)

== Conclusion

- Les couches convolutives sont les briques de base des CNN
- Les couches de pooling réduisent la taille spatiale et apportent une robustesse aux petits décalages
- BatchNorm stabilise l'entraînement et agit comme un régularisateur léger
- L'augmentation de données est une technique de régularisation très efficace pour les images
