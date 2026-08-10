#import "cetz/colors.typ": *

#let image-with-caption(image, caption, image-align: center, caption-align: right) = {
  stack(
    spacing: 6pt,
    align(
      image-align,
      image
    ),
    align(
      caption-align,
      text(
        size: 0.8em,
        caption,
      ),
    ),
  )
}

#let figure-placeholder(
  width,
  height,
  legend: none,
) = {
  let box = rect(
    width: width,
    height: height,
    stroke: 1pt,
    inset: 0pt,
  )

  if legend == none {
    box
  } else {
    image-with-caption(box, legend)
  }
}

#let algo-box(title, body, color: rgb(131,109,169)) = {
  rect(
    width: 100%,
    inset: 10pt,
    radius: 4pt,
    stroke: 0pt + color,
  )[
    #text(weight: "bold", fill: color)[#title]
    #v(4pt)
    #body
  ]
}

#let grad-disk(
  size: .8em,
  from: xi,
  to: yj,
  angle: 0deg,
) = {
  box(
    width: size,
    height: size,
    baseline: 0%,
  )[
    #let grad = gradient.linear(
      from,
      to,
      angle: angle
    )
    #circle(
      radius: size / 2,
      fill: grad.sharp(2),
    )
  ]
}

#let cite-as(cite-key, ref-key, display-text) = {
  cite(cite-key, form: none)
  link(ref-key, display-text)
}
