#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Generative neural networks],
  config-info(
    title: [Deep Learning],
    subtitle: [6. Generative neural networks],
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

== Generative models in a nutshell

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - Goal: model $p(x)$ or $p(x|y)$
      - explicitly
        - _eg._ Gaussian Mixture Models
      - *implicitly*
        - at least allow for sampling (_i.e._ #underline[generate] new data)
        - _e.g._ Variational Auto Encoders, Generative Adversarial Networks
  ],
  [
    #image-with-caption(image("fig/gmm_2d.svg", width: 100%), [_GMM with 2 components_])
  ]
)

== Auto-encoders [Hinton & Salakhutdinov, 2006]

- Encode information in a latent space
  - typically lower-dimensional
  - similar in spirit to a non-linear PCA
  - not a generative model _per se_!

#image-with-caption(image("fig/autoencoder.png", width: 80%), [Source: lilianweng.github.io])

== Variational auto-encoders [Kingma & Welling, 2014]

- *Goal:* turn auto-encoders into generative models
- *Idea:* prior $p(bold(z)) = cal(N)(0, bold(I))$ on latent space, enforced by KL:
  $ cal(L) = underbrace(- log p_theta (bold(x)|bold(z)), "reconstruction") + underbrace("KL"(q_Phi (bold(z)|bold(x)) || p(bold(z))), "stay close to prior") $
- *Generative process:* draw z from prior → decode

#image-with-caption(image("fig/vae-gaussian.png", width: 80%), [Source: lilianweng.github.io])

== GANs [Goodfellow _et al._, 2014]

#image-with-caption(image("fig/GAN.png", width: 70%), [_Source: KDNuggets.com_])

- Two networks compete
  - $D$ (discriminator): outputs P(sample is real)
  - $G$ (generator): turns noise → fakes
  - $D$ wants to be right, $G$ wants to fool $D$ → *min-max*

#pagebreak()

- Loss function:
  $
    min_G max_D L(D, G) = underbrace(EE_(x tilde p_r (x)) [log D(x)], "D right on real data") + underbrace(EE_(z tilde p_z (z)) [log(1 - D(G(z)))], "D right on fakes / G wants this small")
  $
  - $max_D$: pushes both terms up (sharper discriminator)
  - $min_G$: pushes 2nd term down ($D(G(z)) arrow.r 1$, fakes look real)
- Optimization
  - Alternate between training $G$ and $D$
  - Very unstable in practice
- Generative process
  - Draw $z$ from $p_z$ (random noise)
  - Pass it to the generator to compute $G(z)$ (fake sample)

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
    - *Principle:* learn a vector field transporting noise ($t$=0) → data ($t$=1) by minimizing
      $ EE_(x_0, x_1, t) || u^theta (x_t, t) - (x_1 - x_0) ||^2 $
    - At generation time, use an Ordinary Differential Equation solver (eg. Euler scheme):
      $ x_(t + epsilon) arrow.l x_t + epsilon u^theta (x_t, t) $
  ],
  [
    #image-with-caption(image("fig/cfm.svg", width: 100%), [_Learned velocity field transports noise (t=0) to data (t=1). \ Source: Emonet et al. "A Visual Dive into Conditional Flow Matching", ICLR 2025_])
  ]
)
