#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "tools.typ": *

#show: university-theme.with(
  aspect-ratio: "4-3",
  align: horizon,
  footer-b: [Deep Learning - Entraîner un modèle avec `torch`],
  config-info(
    title: [Deep Learning],
    subtitle: [3b. Entraîner un modèle avec `torch`],
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
#show raw.where(block: true): set text(size: 0.75em)
#show raw.where(block: true): block.with(
  width: 100%,
  inset: 8pt,
  radius: 4pt,
  fill: rgb(245,243,249),
)

#title-slide()

== Les 4 ingrédients d'un entraînement

- *Un modèle* : une fonction paramétrée $m(x; theta)$ \
  → `torch.nn.Module`
- *Une fonction de perte* : mesure de l'erreur $cal(L)$ \
  → `torch.nn.MSELoss`, `torch.nn.CrossEntropyLoss`, ...
- *Des données*, servies par _mini-batches_ \
  → `torch.utils.data.Dataset` + `DataLoader`
- *Un optimiseur* : applique les mises à jour de $theta$ \
  → `torch.optim.SGD`, `torch.optim.Adam`, ...

// = Le modèle : `nn.Module`

== Définir un modèle : `nn.Module`

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)   # pas d'activation de sortie ici

model = MLP(n_in=13, n_hidden=64, n_out=1)
y_hat = model(x)             # appelle model.forward(x)
```

- `__init__` : *déclarer* les couches (et donc les paramètres)
- `forward` : *décrire le calcul* de la sortie à partir de l'entrée

// == `nn.Module` : ce que l'on obtient gratuitement

// - *Enregistrement automatique des paramètres*
//   ```python
//   for name, p in model.named_parameters():
//       print(name, p.shape)   # fc1.weight (64, 13), fc1.bias (64,), ...
//   ```
// - *Raccourci* pour les architectures séquentielles
//   ```python
//   model = nn.Sequential(
//       nn.Linear(13, 64), nn.ReLU(),
//       nn.Linear(64, 1),
//   )
//   ```
// - *Modes* `model.train()` / `model.eval()` \
//   (utile pour Dropout, BatchNorm : cf. séance 4)
// - *Déplacement* vers GPU : `model.to("cuda")`

== Fonctions de perte en `torch`

- Une perte est aussi un `nn.Module` : on l'instancie, puis on l'appelle
  ```python
  loss_fn = nn.MSELoss()            # régression
  loss = loss_fn(y_hat, y)          # scalaire (moyenne sur le batch)
  ```
- *Classification* : `nn.CrossEntropyLoss`
  ```python
  loss_fn = nn.CrossEntropyLoss()
  logits = model(x)                 # (batch, n_classes), SANS softmax
  loss = loss_fn(logits, y)         # y : indices de classes (batch,)
  ```
  - ⚠ le softmax est *inclus* dans la perte : \
    pas de softmax en sortie du modèle
  - cas binaire : `nn.BCEWithLogitsLoss` (sigmoïde incluse)

== `Dataset` et `DataLoader`

- `Dataset` : *accès à un exemple* `(x_i, y_i)` par son indice
  ```python
  from torch.utils.data import TensorDataset, DataLoader

  X = torch.tensor(X_np, dtype=torch.float32)
  y = torch.tensor(y_np, dtype=torch.float32)
  dataset = TensorDataset(X, y)     # dataset[i] -> (X[i], y[i])
  ```
- `DataLoader` : *découpage en _mini-batches_*
  ```python
  loader = DataLoader(dataset, batch_size=32, shuffle=True)

  for x_batch, y_batch in loader:   # un tour de boucle = un mini-batch
      ...                           # un tour complet = une epoch
  ```
  - `shuffle=True` : nouvel ordre à chaque _epoch_ (SGD, cf. séance 2)
  - pour la validation / le test : `shuffle=False`

// == `Dataset` personnalisé

// - Utile quand les données ne tiennent pas en mémoire (images sur disque, ...)

// ```python
// from torch.utils.data import Dataset

// class MyDataset(Dataset):
//     def __init__(self, paths, labels):
//         self.paths, self.labels = paths, labels

//     def __len__(self):
//         return len(self.paths)

//     def __getitem__(self, i):
//         x = load_image(self.paths[i])   # chargement à la demande
//         return x, self.labels[i]
// ```

// - Le `DataLoader` se charge de regrouper les exemples en _batches_

// = Assembler le tout

// == La boucle d'entraînement

// ```python
// model = MLP(n_in=13, n_hidden=64, n_out=1)
// loss_fn = nn.MSELoss()
// optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
// loader = DataLoader(dataset, batch_size=32, shuffle=True)

// for epoch in range(n_epochs):
//     model.train()
//     for x_batch, y_batch in loader:
//         optimizer.zero_grad()              # 1. remise à zéro des gradients
//         y_hat = model(x_batch)             # 2. passe avant
//         loss = loss_fn(y_hat, y_batch)     # 3. calcul de la perte
//         loss.backward()                    # 4. rétropropagation
//         optimizer.step()                   # 5. mise à jour de theta
// ```

// - Les étapes 1 à 5 correspondent à *une itération de SGD*

// == Évaluer le modèle

// ```python
// model.eval()                         # mode évaluation
// with torch.no_grad():                # pas de calcul de gradient
//     total = 0.
//     for x_batch, y_batch in test_loader:
//         total += loss_fn(model(x_batch), y_batch).item() * len(x_batch)
//     print(total / len(test_loader.dataset))
// ```

// - `torch.no_grad()` : plus rapide, moins de mémoire
// - `.item()` : tenseur scalaire → nombre Python

== Conclusion

#grid(
  columns: (auto, 1fr),
  gutter: 1em,
  row-gutter: 1.2em,
  [*`nn.Module`*], [architecture : couches dans `__init__`, calcul dans `forward`],
  [*Perte*], [un `nn.Module` qui compare prédictions et cibles \ (⚠ `CrossEntropyLoss` attend des _logits_)],
  [*`DataLoader`*], [itère sur les données par _mini-batches_ mélangés],
  [*Optimiseur*], [`zero_grad()` → `backward()` → `step()`],
)
