#import "@preview/cetz:0.5.2"
#import "colors.typ": *

#let ink = rgb(43, 36, 56)
#let purple = rgb(131, 109, 169)
#let batch-colors = (xi, yj, newlink)

// Text size is pinned in absolute pt so the diagram doesn't get thrown out
// of proportion by whatever ambient font size the including slide uses.
#set text(size: 11pt, fill: ink)

#cetz.canvas({
  import cetz.draw: *
  set-style(stroke: ink)

  let n = 12
  let m = 4
  let nb = calc.quo(n, m)  // number of mini-batches
  let sz = .6     // side of one sample square
  let step = .7   // distance between two consecutive samples
  let gap = .5    // extra gap between two mini-batches
  let batch-w = (m - 1) * step + sz
  let batch-x(k) = k * (batch-w + gap + step - sz)
  let sample-x(i) = batch-x(calc.floor(i / m)) + calc.rem(i, m) * step
  let total-w = batch-x(nb - 1) + batch-w
  let mid = total-w / 2

  // One row of samples; `perm` gives the sample indices in visiting order.
  let samples(y, perm, colored: true) = {
    for (i, idx) in perm.enumerate() {
      let x = if colored { sample-x(i) } else { i * (total-w - sz) / (n - 1) }
      let c = if colored { batch-colors.at(calc.floor(i / m)) } else { activeset }
      rect((x, y), (x + sz, y + sz), fill: c.lighten(if colored { 55% } else { 0% }),
        stroke: (if colored { c } else { activesetfont }) + .8pt, radius: .06)
      content((x + sz / 2, y + sz / 2), text(size: 8pt)[#idx])
    }
  }

  // An epoch: shuffled samples, mini-batch braces, chain of weight updates.
  let epoch(y, perm, label, t0) = {
    samples(y, perm)
    content((-.5, y + sz / 2), text(fill: purple, weight: "bold")[#label], anchor: "east")
    for k in range(nb) {
      let x0 = batch-x(k)
      cetz.decorations.flat-brace((x0 + batch-w, y - .1), (x0, y - .1),
        stroke: batch-colors.at(k) + .8pt, amplitude: .25)
      content((x0 + batch-w / 2, y - .6), text(size: 9pt, fill: batch-colors.at(k))[$B_#(k + 1)$])
    }
    let wy = y - 1.3
    for k in range(nb + 1) {
      let x = if k == 0 { -.15 } else { batch-x(k) - gap / 2 - (step - sz) / 2 }
      content((x, wy), [$w^((#(t0 + k)))$], name: "w" + str(k))
    }
    for k in range(nb) {
      line("w" + str(k), "w" + str(k + 1), mark: (end: ">", fill: ink), stroke: ink + .8pt)
    }
  }

  // --- full data set, in its original order
  let y-data = 2.2
  samples(y-data, range(1, n + 1), colored: false)
  content((-.5, y-data + sz / 2), [Données], anchor: "east")
  line((mid, y-data - .15), (mid, .85),
    mark: (end: ">", fill: purple), stroke: purple + 1pt)
  content((mid + .2, 1.4),
    text(size: 9pt, fill: purple)[mélange + découpage en _mini-batches_ ($m = #m$)],
    anchor: "west")

  // --- two epochs, each with its own shuffle
  epoch(0, (7, 2, 11, 5, 9, 1, 12, 4, 3, 10, 6, 8), [Epoch 1], 0)
  epoch(-3.2, (4, 9, 1, 6, 12, 3, 8, 2, 10, 5, 7, 11), [Epoch 2], 3)
  line((mid, -1.65), (mid, -2.45), mark: (end: ">", fill: purple), stroke: purple + 1pt)
  content((mid + .2, -2.05), text(size: 9pt, fill: purple)[nouveau mélange], anchor: "west")
})
