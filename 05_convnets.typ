#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Images and Convolutional neural networks],
  config-info(
    title: [Deep Learning],
    subtitle: [5. Images and Convolutional neural networks],
    author: [Romain Tavenard],
    date: [
      #align(bottom)[
        #text(size: 0.8em)[NB: Most figures in these slides are from _Dumoulin & Visin. "A guide to convolution arithmetic for deep learning". 2016_]
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

== Preamble: What's an image?

#image-with-caption(image("fig/rgb_pixels.svg", width: 100%), [])

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
    #align(center)[
      #image("fig/conv_no_padding_no_strides.svg", width: 70%) \
      #v(-3em)#image("fig/conv_no_pad.svg", width: 100%)
    ]
  ]
)

== Convolutional layers in NN (1/2)

- A convolution layer is made of
  - convolution kernels
  - biases (1 per kernel)
  - an activation function

- Useful because
  - reduces \#parameters (*parameter sharing*: same kernel reused everywhere)
    - _e.g._ dense neuron on $224 times 224$: $approx 50 000$ weights vs $3 times 3$ kernel: 9
  - encodes *translation equivariance* (translation in the input induces translation in the output)

// == Convolution and translation

// #figure-placeholder(100%, 300pt, legend: [Figure: A "4" digit image on left, passed through a conv layer (green), output is a stack of feature maps showing translated activation patterns. Source: Christian Wolf, Twitter https://twitter.com/chriswolfvision/status/1313059518574718977?s=20])

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
    #image-with-caption(image("fig/conv_multichannel.svg", width: 110%), [])
  ]
)

== Convolutional layers in NN: hyper parameters

- Padding

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

- Strides

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

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - *Why pool?*
      - shrink spatial size → fewer computations
      - robustness to small shifts (conv = equivariant, pooling → invariant)
    - Max pooling / Average pooling
    - Hyper-parameter
      - pool size
  ],[
    #image-with-caption(image("fig/pool_max.svg", width: 100%), [_Max pooling example_])
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
    MNIST dataset \
    10 classes \
    60,000 images
  ]
)

#image-with-caption(image("fig/LeNet5.svg", width: 100%), [_LeNet-5 architecture (LeCun et al., 1989)_. Source: Wikipedia])

== A drastic improvement on performance (ImageNet)

- ImageNet
  - 15M images
  - 22k classes
- LSVRC: Subset of ImageNet (1.2M images, 1k classes)
#image-with-caption(image("fig/lsvrc.svg", width: 75%), [])

== Regularization: Data Augmentation

- Recall : big nets over-fit → regularize (early stopping, Dropout, BatchNorm)
- Images: one extra, very effective trick: *Data Augmentation*
- Principle: generate virtual training examples
  // - original image $x_i$
  // - modified image $hat(x)_i$
  // - unchanged label $y_i$

#grid(
  columns: (.7fr, 1fr),
  gutter: 1em,
  align: bottom,
  [
    #image-with-caption(image("fig/aug_original.svg", width: 75%), [_Original image_ (label: "cat")], image-align: center, caption-align: center)
  ],
  [
    #image-with-caption(image("fig/aug_grid.svg", width: 100%), [_6 augmented versions (same label)_], caption-align: center)
  ]
)