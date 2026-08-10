#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Generative neural networks],
  config-info(
    title: [Deep Learning],
    subtitle: [5. Generative neural networks],
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
    #image-with-caption(image("fig/gmm.svg", width: 100%), [_GMM: $p(x)$ = sum of Gaussian components_])
  ]
)

== Auto-encoders #linebreak() [Hinton & Salakhutdinov, 2006]

- Encode information in a latent space
  - typically lower-dimensional
  - similar in spirit to a non-linear PCA
  - not a generative model _per se_!

#figure-placeholder(85%, 200pt, legend: [_Figure: Auto-encoder diagram — Input *x* (digit "4") → Encoder $g_phi$ → Bottleneck *z* (compressed representation) → Decoder $f_theta$ → Reconstructed input *x'* (digit "4"). Ideally $bold(x) approx bold(x)'$. Source: lilianweng.github.io_])

== Variational auto-encoders #linebreak() [Kingma & Welling, 2014]

- #underline[Goal:] turn auto-encoders into generative models
- #underline[Idea:] prior $p(bold(z)) = cal(N)(0, bold(I))$ on latent space, enforced by KL penalty:
  $ cal(L) = underbrace(- log p_theta (bold(x)|bold(z)), "reconstruction") + underbrace("KL"(q_phi (bold(z)|bold(x)) || p(bold(z))), "stay close to prior") $
- #underline[Generative process:] draw z from prior → decode

#figure-placeholder(85%, 230pt, legend: [_Figure: VAE diagram — Input *x* → Probabilistic Encoder $q_phi (bold(z)|bold(x))$ outputting Mean $mu$ and Std. dev $sigma$ → sampled latent vector $bold(z) = mu + sigma dot.o epsilon$, $epsilon tilde cal(N)(0, bold(I))$ → Probabilistic Decoder $p_theta (bold(x)|bold(z))$ → Reconstructed input *x'*. Source: lilianweng.github.io_])

== Generative Adversarial Networks #linebreak() [Goodfellow _et al._, 2014]

- Model:

#figure-placeholder(85%, 160pt, legend: [_Figure: GAN diagram — Latent Space (noise) → Generator G (learns data distribution) + Real Samples → Discriminator D (learns to tell apart fake from real) → Is D correct? → Fine Tune Training loop. $y = cases(1 "for a real sample", 0 "for a fake sample")$. Source: lilianweng.github.io_])

- *Idea first:* two networks compete
  - $D$ (detective): outputs P(sample is real)
  - $G$ (forger): turns noise → fakes
  - $D$ wants to be right, $G$ wants to fool $D$ → *min-max*

- Loss function:

$
  min_G max_D L(D, G) = underbrace(EE_(x tilde p_r (x)) [log D(x)], "D right on real data") + underbrace(EE_(z tilde p_z (z)) [log(1 - D(G(z)))], "D right on fakes / G wants this small")
$
$
  = EE_(x tilde p_r (x)) [log D(x)] + EE_(x tilde p_g (x)) [log(1 - D(x))]
$

- $max_D$: pushes both terms up (sharper detective)
- $min_G$: pushes 2nd term down ($D(G(z)) arrow.r 1$, fakes look real)

== Generative Adversarial Networks #linebreak() [Goodfellow _et al._, 2014]

- *Intuition:* think of a forger (Generator) and a detective (Discriminator)
  - The forger tries to produce fake paintings indistinguishable from real ones
  - The detective tries to tell fakes from real ones
  - They improve each other through competition — the forger gets better because the detective gets better, and vice versa

- Generative process
  - Draw $z$ from $p_z$ (random noise)
  - Pass it to the generator to compute $G(z)$ (fake sample)

- Optimization
  - Alternate between training G and D
  - Very unstable in practice — both players must improve at the same pace

== Generative Adversarial Networks

- Many variants to the original model
  - Class-conditional variants
  - Different losses
  - Different structures
- Very realistic samples generated (BigGAN, StyleGAN)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #figure-placeholder(100%, 200pt, legend: [_Figure: BigGAN butterfly samples — grid of realistic butterfly photos_])
  ],
  [
    #figure-placeholder(100%, 200pt, legend: [_Figure: StyleGAN face samples — two photorealistic synthetic faces_])
  ]
)

== Diffusion models #linebreak() [Ho _et al._, 2020]

- *Intuition:*
  - add noise to image → pure static
  - train net to *undo one noising step*
  - generate: start from noise, denoise step by step

- *Context:*
  - just a *regression* (predict the added noise) → stable training (unlike GAN min-max)
  - price: *many* denoising steps (slow)

#figure-placeholder(100%, 200pt, legend: [_Figure: Comparison of three generative model families — GAN: *x'* → Discriminator D(*x*) → 0/1, and *z* → Generator G(*z*) → *x'*. VAE: *x* → Encoder $q_phi (bold(z)|bold(x))$ → *z* → Decoder $p_theta (bold(x)|bold(z))$ → *x'*. Diffusion model: $bold(x)_0 arrow.r bold(x)_1 arrow.r bold(x)_2 arrow.r dots arrow.r bold(z)$ (forward noising) and reverse denoising. Source: lilianweng.github.io_])

#pagebreak()

#image-with-caption(image("fig/diffusion_strip.svg", width: 100%), [_Forward process: data is gradually turned into pure noise; the model learns to reverse it_])

== Diffusion models #linebreak() [Ho _et al._, 2020]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #figure-placeholder(100%, 160pt, legend: [_Algorithm 1 Training: repeat — sample $x_0$, $t$, $epsilon tilde cal(N)(0,bold(I))$; gradient step on $nabla_theta || epsilon - epsilon_theta(sqrt(overline(alpha)_t) bold(x)_0 + sqrt(1-overline(alpha)_t) epsilon, t) ||^2$_])
  ],
  [
    #figure-placeholder(100%, 160pt, legend: [_Algorithm 2 Sampling: $bold(x)_T tilde cal(N)(0, bold(I))$; for $t = T,...,1$: $bold(z) tilde cal(N)(0,bold(I))$ if $t>1$ else $bold(z)=0$; $bold(x)_(t-1) = 1/sqrt(alpha_t)(bold(x)_t - (1-alpha_t)/sqrt(1-overline(alpha)_t) epsilon_theta(bold(x)_t, t)) + sigma_t bold(z)$; return $bold(x)_0$_])
  ]
)

#figure-placeholder(100%, 110pt, legend: [_Figure: Denoising chain diagram (same as previous slide)_])

#align(right)[Source: [Ho et al., 2020]]

== Conditional Flow Matching #linebreak() [Lipman _et al._, 2023]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - #underline[Principle:] learn a vector field transporting noise ($t$=0) → data ($t$=1)

    - #underline[vs diffusion:]
      - same goal (noise → data)
      - *deterministic* path (velocity field) + ODE solver, not a random denoising chain
      - → *fewer steps*, simpler target

    - Loss

    $ EE_(x_0, x_1, t) || u^theta (x_t, t) - (x_1 - x_0) ||^2 $

    - $x_t = (1-t) dot x_0 + t dot x_1$ (straight line noise → data)
    - velocity $dot(x)_t = x_1 - x_0$ = regression target

    - At generation time, use an Ordinary Differential Equation solver (eg. Euler scheme):

    $ x_(t + epsilon) arrow.l x_t + epsilon u^theta (x_t, t) $
  ],
  [
    #image-with-caption(image("fig/flow_field.svg", width: 100%), [_Learned velocity field transports noise (t=0) to data (t=1)_])
  ]
)

== Wrapping up #linebreak() From these building blocks to today's models

- Same pieces power the systems you hear about:
  - *conv + deep stacks* → image recognition
  - *diffusion / flow matching* → text-to-image (Stable Diffusion, Midjourney)
  - *representations + attention* (not covered) → LLMs
- *Takeaway:* same ingredients (differentiable layers, loss, gradient descent), scaled up
- Most real-world use is *not* training from scratch but *reusing* pretrained models (cf. transfer learning)
