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

  circle((1.75, 1.25), radius: .5, name: "o", stroke: purple + 1.2pt)
  content("o", [$o$])

  content((0, 0), [$+1$], name: "bias")
  content((0, 3), [$x_1$], name: "x1")
  content((0, 1), [$x_D$], name: "xD")
  content((2.9, 1.25), [$a = phi(w^t x + b)$], anchor: "west", name: "a")

  line("bias.east", "o", mark: (end: ">", fill: ink), name: "l-bias")
  content((name: "l-bias", anchor: 45%), text(size: 9pt)[$b$], anchor: "north-west")

  line("x1.east", "o", mark: (end: ">", fill: ink), name: "l-x1")
  content((name: "l-x1", anchor: 45%), text(size: 9pt)[$w_1$], anchor: "south-west")

  line("xD.east", "o", mark: (end: ">", fill: ink), name: "l-xD")
  content((name: "l-xD", anchor: 60%), text(size: 9pt)[$w_D$], anchor: "south-east")

  line("o", "a.west", mark: (end: ">", fill: ink), name: "l-out")
  content((name: "l-out", anchor: 30%), text(size: 9pt)[$phi$], anchor: "south")

  line("x1", "xD", stroke: (paint: ink, dash: "dashed"))
})
