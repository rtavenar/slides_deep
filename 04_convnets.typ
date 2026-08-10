#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Images and Convolutional neural networks],
  config-info(
    title: [Deep Learning],
    subtitle: [4. Images and Convolutional neural networks],
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

#align(bottom)[
  #text(size: 0.8em)[NB: Most figures in these slides are from \ Dumoulin & Visin. A guide to convolution arithmetic for deep learning. 2016]
]

== Preamble: What's an image?

#figure-placeholder(100%, 310pt, legend: [_Figure: RGB cat image split into 3 colored channel planes, plus a 3D tensor schematic labeled height × width × channel. Source: "Understanding Images with skimage-Python", Towards Data Science_])

== Convolution in practice

#figure-placeholder(100%, 320pt, legend: [Figure: Two-panel image — left: pixel grid of Mario sprite, right: a filtered/convolved version. Source: Grant Sanderson, Twitter https://twitter.com/3blue1brown/status/1303489896519139328?s=20])

== The convolution operator

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - 2D convolution
      - Blue: input image
      - Gray: convolution kernel
      - Cyan: activation map

    - Convolution operation = Dot product between
      - convolution kernel (aka filter)
      - subpart of the input
  ],
  [
    #image-with-caption(image("fig/conv_no_pad.svg", width: 100%), [_Blue input, shaded kernel position → one cyan output cell_])
  ]
)

== Convolutional layers in NN (1/2)

- A convolution layer is made of
  - convolution kernels
  - biases (1 per kernel)
  - an activation function

- Useful because
  - reduces \#parameters (*parameter sharing*: same kernel reused everywhere)
    - _e.g._ dense neuron on $224 times 224$: $approx 50 000$ weights vs $3 times 3$ kernel: *9*
  - encodes translation equivariance (translation in the input induces translation in the output, cf. next slide)

== Convolution and translation

#figure-placeholder(100%, 300pt, legend: [Figure: A "4" digit image on left, passed through a conv layer (green), output is a stack of feature maps showing translated activation patterns. Source: Christian Wolf, Twitter https://twitter.com/chriswolfvision/status/1313059518574718977?s=20])

== Convolutional layers in NN (2/2)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - Multiple input channel case
      - sum the response over all channels
    - Multiple kernel case
      - each kernel leads to one output channel
  ],
  [
    #figure-placeholder(100%, 240pt, legend: [_Figure: Tree diagram showing 1 input (bottom), 2 input channels feeding into 3 kernels, each producing an output channel, summed to top output. Caption: "2 input channels, 3 kernels"_])
  ]
)

== Convolutional layers in NN: hyper parameters

- Padding

#grid(
  columns: (1fr, 1fr),
  gutter: 0.5em,
  [
    #image("fig/conv_same.svg", width: 100%)
  ],
  [
    #image("fig/conv_valid.svg", width: 100%)
  ]
)

- Strides

#grid(
  columns: (1fr, 1fr),
  gutter: 0.5em,
  [
    #image("fig/conv_stride1.svg", width: 100%)
  ],
  [
    #image("fig/conv_stride2.svg", width: 100%)
  ]
)

== Computing the size of an activation map

- Assumption: no padding ("valid"), unit strides

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #image-with-caption(image("fig/conv_valid.svg", width: 100%), [_Output shrinks by kernel size − 1_])
  ],
  [
    $
      W_"out" = W_"in" - W_k + 1
    $
    $
      H_"out" = H_"in" - H_k + 1
    $

    *Example:*
    - Input: $28 times 28$ (MNIST digit)
    - Kernel: $5 times 5$
    - Output: $(28-5+1) times (28-5+1) = 24 times 24$

    *With padding="same":* output stays $28 times 28$
  ]
)

== Pooling (aka subsampling) layers in NN

- *Why pool?*
  - shrink spatial size → fewer computations
  - robustness to small shifts (conv = equivariant, pooling → invariant)
- Max pooling / Average pooling
- Hyper-parameters
  - pool size
  - strides (use None in keras)
  - padding (use "valid" in keras)

#figure-placeholder(50%, 140pt, legend: [_Figure: max pooling — each pool region keeps its strongest activation_])

= A history of Convolutional neural networks (CNN)

== A history of CNN #linebreak() LeCun (1989)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #figure-placeholder(100%, 160pt, legend: [_Figure: MNIST digit grid_])
  ],
  [
    MNIST dataset \
    10 classes \
    60,000 images
  ]
)

#figure-placeholder(100%, 120pt, legend: [_Figure: LeNet-5 architecture diagram — input 32×32 → C1 (feature maps 28×28) → S1 (14×14) → C2 (10×10) → S2 (5×5) → n1 → n2 → output, with 5×5 convolution and 2×2 subsampling labels_])

== A history of CNN #linebreak() Deeper and deeper networks

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - Deeper = higher-level understanding
    - More depth → more expressive power
    - ...but *vanishing gradients* (cf. ch. 3)
    - Mitigated by: ReLU, init, BatchNorm, residual connections (later)
  ],
  [
    #align(center)[#image("fig/mlp_deep.svg", width: 100%)]
  ]
)

== A history of CNN #linebreak() ImageNet & LSVRC (2012)

- ImageNet
  - 15M images
  - 22k classes
- LSVRC
  - Subset of ImageNet (1.2M images, 1k classes)

#figure-placeholder(100%, 180pt, legend: [_Figure: ImageNet taxonomy visualization — mammal → placental → carnivore → canine → dog → working dog → husky, and vehicle → craft → watercraft → sailing vessel → sailboat → trimaran_])

== A history of CNN #linebreak() A drastic improvement on performance (LSVRC)

#image-with-caption(image("fig/lsvrc.svg", width: 75%), [_Deep nets (purple) crush shallow ones (orange) from 2012 on_])

== A history of CNN #linebreak() AlexNet (2012)

#figure-placeholder(100%, 160pt, legend: [_Figure: AlexNet architecture diagram — Input 224×224×3 → conv stride 4, 96 filters → max pool → conv 256 → max pool → conv 384 → conv 384 → conv 256 → max pool → FC 4096 → FC 4096 → FC 1000_])

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - Error rate: 15%
    - 60M parameters
    - 2 GPUs -- 6 days
  ],
  [
    - Regularization
      - Data augmentation
      - Dropout
      - L2
  ]
)

== A history of CNN #linebreak() What does AlexNet learn?

Sample convolution filters learned:

#figure-placeholder(100%, 220pt, legend: [_Figure: Visualization of learned conv filters for AlexNet layers 1–5. Layer 1: simple edges/colors; layers 2–5: increasingly abstract patterns. Source: [Zeiler & Fergus, 2013]_])

== Over-parametrization in deep learning #linebreak() One more regularizer for images

- Recall (ch. 3): big nets over-fit → regularize (L2, early stopping, Dropout, BatchNorm)
- For *images*: one extra, very effective trick
  - *Data Augmentation* — next

== Regularization: Data Augmentation

- Principle: generate virtual training examples
  - original image $x_i$
  - modified image $hat(x)_i$
  - unchanged label $y_i$

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #figure-placeholder(100%, 180pt, legend: [_Figure: Original kitten $x_i$ on left_])
  ],
  [
    $y_i = $ "cat"

    #figure-placeholder(100%, 180pt, legend: [_Figure: Grid of 6 augmented $hat(x)_i$ versions (flips, crops, rotations). Image from nanonets.com_])
  ]
)

== A history of CNN #linebreak() _ResNet_ (2015) — going really deep

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - *Problem:* past ~20 layers, deeper = *worse* (vanishing gradients → early layers un-trainable)
    - *Intuition (skip connection):* block learns a *correction*, not a full transform:
      $ bold(y) = bold(x) + cal(F)(bold(x)) $
    - nothing useful to add → output $bold(x)$ → depth never hurts
    - enabled *100+ layers*; won ILSVRC 2015
  ],
  [
    #figure-placeholder(100%, 200pt, legend: [_Figure: Residual block — input *x* splits into a conv path $cal(F)(bold(x))$ and an identity skip connection, summed at the output $bold(x) + cal(F)(bold(x))$_])
  ]
)

#pagebreak()

#text(weight: "bold")[Why the skip connection fixes vanishing gradients]

- The gradient flowing back through one residual block is:

$
  (partial bold(y))/(partial bold(x)) = underbrace(1, "identity shortcut") + (partial cal(F)(bold(x)))/(partial bold(x))
$

- the *"1" is key*: even if $partial cal(F) slash partial bold(x)$ tiny, gradient $approx 1$ (not $approx 0$)
- stacking $L$ blocks → product of $(1 + dots)$ terms → *no collapse to zero*

== Transfer learning #linebreak() Don't train from scratch!

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - *Key idea:* CNN trained on ImageNet already learned generic features (edges → textures → parts)
    - *Reuse* on your (small) dataset:
      - keep pretrained conv layers ("backbone")
      - replace + train only the final classifier
      - optionally *fine-tune* top layers (small LR)
    - *Why:* strong results, little data/compute → most common real-world use
  ],
  [
    #figure-placeholder(100%, 240pt, legend: [_Figure: Pretrained backbone (frozen conv layers) + new task-specific head being trained on a small dataset_])

    ```python
    from keras.applications import ResNet50

    base = ResNet50(
        include_top=False,
        weights="imagenet")
    base.trainable = False  # freeze
    ```
  ]
)
