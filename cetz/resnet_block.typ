#import "@preview/cetz:0.5.2"

#let ink = rgb(43, 36, 56)
#let purple = rgb(131, 109, 169)

// Text size is pinned in absolute pt so the diagram doesn't get thrown out
// of proportion by whatever ambient font size the including slide uses.
#set text(size: 11pt, fill: ink)

#cetz.canvas({
  import cetz.draw: *
  set-style(
    content: (padding: .1),
    stroke: ink,
  )

  // --- main trunk: x -> weight layer -> ReLU -> weight layer -> sum -> ReLU -> y
  content((1, 8), [$bold(x)$], name: "x")

  rect((0.15, 6), (1.85, 7), name: "conv1", stroke: purple + 1.2pt, radius: .08)
  content("conv1", text(size: 8.5pt)[Conv. layer])

  content((1, 5.2), text(size: 9pt)[ReLU], name: "relu1")

  rect((0.15, 3.5), (1.85, 4.5), name: "conv2", stroke: purple + 1.2pt, radius: .08)
  content("conv2", text(size: 8.5pt)[Conv. layer])

  circle((1, 2.6), radius: .3, name: "sum", stroke: purple + 1.4pt)
  content("sum", text(size: 14pt)[$+$])

  content((1, 1.5), [$bold(y) = bold(x) + cal(F)(bold(x))$], name: "y")
  content((1, 0.5), text(size: 9pt)[ReLU], name: "relu2")

  rect((0, 3.3), (2, 7.4), name: "F", stroke:  (paint: purple, thickness: 1.2pt, dash: "dashed"), radius: .08)
  content((-.1, 5.2), text(size: 9.5pt, fill: purple)[$cal(F)$],
    anchor: "east")
  
  line("x", "conv1", mark: (end: ">", fill: ink))
  line("conv1", "relu1", mark: (end: ">", fill: ink))
  line("relu1", "conv2", mark: (end: ">", fill: ink))
  line("conv2", "sum", mark: (end: ">", fill: ink))
  line("sum", "y", mark: (end: ">", fill: ink))
  line("y", "relu2", mark: (end: ">", fill: ink))

  // --- skip connection: identity path from x straight to the sum node
  line("x.east", (2.5, 8), (2.5, 2.6), "sum.east",
    stroke: purple + 1.4pt, mark: (end: ">", fill: purple))
  content((2.65, 5.2), text(size: 9.5pt, fill: purple)[skip connection \ (identity)],
    anchor: "west")
})
