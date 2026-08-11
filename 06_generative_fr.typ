#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Réseaux de neurones génératifs],
  config-info(
    title: [Deep Learning],
    subtitle: [6. Réseaux de neurones génératifs],
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

== Les modèles génératifs en un coup d'œil

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - Objectif : modéliser $p(x)$ ou $p(x|y)$
      - explicitement
        - _ex._ modèles de mélange gaussien
      - *implicitement*
        - au moins permettre l'échantillonnage (_c'est-à-dire_ #underline[générer] de nouvelles données)
        - _ex._ auto-encodeurs variationnels, réseaux antagonistes génératifs
  ],
  [
    #image-with-caption(image("fig/gmm_2d_fr.svg", width: 100%), [_GMM à 2 composantes_])
  ]
)

== Auto-encodeurs [Hinton & Salakhutdinov, 2006]

- Encodent l'information dans un espace latent
  - typiquement de dimension plus faible
  - dans le même esprit qu'une ACP non linéaire
  - pas un modèle génératif _per se_ !

#image-with-caption(image("fig/autoencoder.png", width: 80%), [Source : lilianweng.github.io])

== Auto-encodeurs variationnels [Kingma & Welling, 2014]

- Transformer les auto-encodeurs en modèles génératifs
- Une loi $p(bold(z)) = cal(N)(0, bold(I))$ sur l'espace latent, imposée par :
  $ cal(L) = underbrace(- log p_theta (bold(x)|bold(z)), "reconstruction") + underbrace("KL"(q_Phi (bold(z)|bold(x)) || p(bold(z))), "rester proche de l'a priori") $
- *Processus génératif :* tirer z depuis l'a priori → décoder

#image-with-caption(image("fig/vae-gaussian.png", width: 80%), [Source : lilianweng.github.io])

== Les GAN [Goodfellow _et al._, 2014]

#image-with-caption(image("fig/GAN.png", width: 70%), [_Source : KDNuggets.com_])

- Deux réseaux s'affrontent
  - $D$ (discriminateur) : produit P(l'échantillon est réel)
  - $G$ (générateur) : transforme du bruit en faux échantillons
  - $D$ veut avoir raison, $G$ veut tromper $D$ → *min-max*

#pagebreak()
#v(-.8em)
- Fonction de perte :
  $
    min_G max_D L(D, G) = underbrace(EE_(x tilde p_r (x)) [log D(x)], "D a raison (réel)") + underbrace(EE_(z tilde p_z (z)) [log(1 - D(G(z)))], "D a raison (faux) / G veut ↓")
  $
  - $max_D$ : pousse les deux termes vers le haut (discriminateur plus tranchant)
  - $min_G$ : pousse le 2e terme vers le bas ($D(G(z)) arrow.r 1$, les faux semblent réels)
- Optimisation : Alterner l'entraînement de $G$ et de $D$
  - Très instable en pratique
- Processus génératif
  - Tirer $z$ selon $p_z$ (bruit aléatoire)
  - Le passer au générateur pour calculer $G(z)$ (échantillon factice)

// == Diffusion models [Ho _et al._, 2020]

// - *Intuition:*
//   - add noise to image → pure static
//   - train net to *undo one noising step*
//   - generate: start from noise, denoise step by step

// - *Context:*
//   - just a *regression* (predict the added noise) → stable training (unlike GAN min-max)
//   - price: *many* denoising steps (slow)

// #pagebreak()

// #image-with-caption(image("fig/diffusion_strip.svg", width: 100%), [_Forward process: data is gradually turned into pure noise; \ the model learns to reverse it_])

== Conditional Flow Matching [Lipman _et al._, 2023]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - *Principe :* apprendre un champ de vecteurs transportant le bruit ($t$=0) vers les données ($t$=1) en minimisant
      $ EE_(x_0, x_1, t) || u^theta (x_t, t) - (x_1 - x_0) ||^2 $
    - Au moment de la génération, utiliser un solveur d'équation différentielle ordinaire (par ex. schéma d'Euler) :
      $ x_(t + epsilon) arrow.l x_t + epsilon u^theta (x_t, t) $
  ],
  [
    #image-with-caption(image("fig/cfm.svg", width: 100%), [_Le champ de vitesse appris transporte le bruit (t=0) vers les données (t=1). \ Source : Emonet et al. "A Visual Dive into Conditional Flow Matching", ICLR 2025_])
  ]
)
