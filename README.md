# orange3-nnviewer

A minimal Orange Data Mining add-on with one widget:
**Neural Network Viewer** — takes the *Model* output of Orange's
Neural Network widget (scikit-learn MLP) and draws the network on a
native Qt graphics scene:

- one column of nodes per layer (inputs, hidden layers, output),
- feature names on the input neurons, target/class names on the output,
- edge width proportional to |weight|, blue = positive, red = negative,
- the numeric weight printed beside each edge (for small networks),
- biases printed under the neurons they feed,
- **hover**: an edge thickens, fills its two end nodes with its colour
  and tints the associated labels; its tooltip shows
  "source → target, weight = …", node tooltips show name and bias.

The diagram always scales to fit the window — there is no manual zoom
or panning.

## Build & install

Build a wheel (a single self-contained plugin file):

    cd orange3-nnviewer
    uv build            # -> dist/orange3_nnviewer-<ver>-py3-none-any.whl

Install it into the same Python environment that runs Orange
(`--no-deps`: Orange's own dependencies are already present, and
re-resolving them can trigger source builds against the bundled
framework Python):

    uv pip install --python <python-that-runs-orange> --no-deps dist/orange3_nnviewer-*.whl

(For the macOS Orange.app bundle, the interpreter is
`/Applications/Orange.app/Contents/MacOS/python`.)

During development, an editable install is more convenient (edits are
live on Orange restart):

    uv pip install --python <python-that-runs-orange> -e .

Then start (or restart) Orange Canvas — the widget appears in the toolbox
under the "Neural Networks" category, named "NN Viewer":

    python -m Orange.canvas

## Publish to PyPI

Once published, the widget is installable from Orange's own UI
(*Options → Add-ons… → Add more… → orange3-nnviewer*):

    uv build
    uv publish          # needs a PyPI token (UV_PUBLISH_TOKEN or --token)

## Test outside Canvas

Trains an MLP on the bundled housing dataset and opens the widget alone:

    python -m orangecontrib.nnviewer.widgets.ow_nn_viewer

## Tests

Run with the Python that runs Orange (Qt renders offscreen, no display
needed):

    python -m pytest -q

## Use in a workflow

    File  →  Neural Network  →  Neural Network Viewer
                          Model ─┘

## Knobs to tweak (in ow_nn_viewer.py)

- `MAX_LABELED_EDGES` — above this edge count, numbers are suppressed
  (tooltips still show every value).
- weight label format — `f"{weight:.2f}"` in `_draw`.
- colours — `POS_COLOR` / `NEG_COLOR`.
- edge width scale — `1.0 + 4.0 * abs(weight) / max_abs`.
- node size — `radii = [min(0.05, 0.3 * g) for g in y_gaps]`.
- label placement — `t = 0.60/0.40` along the line in `_draw`.