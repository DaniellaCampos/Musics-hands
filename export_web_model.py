"""Exporta hand_model.pt a web/model.json para ejecutar el MLP en el navegador."""
import argparse
import json

from torch import nn

from common import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hand_model.pt")
    parser.add_argument("--out", default="web/model.json")
    args = parser.parse_args()

    model, classes = load_model(args.model)
    rnd = lambda xs: [round(float(v), 6) for v in xs]
    layers = [{"W": [rnd(row) for row in m.weight.tolist()], "b": rnd(m.bias.tolist())}
              for m in model.net if isinstance(m, nn.Linear)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "layers": layers}, f)
    print(f"{args.out}: {len(layers)} capas, clases={classes}")


if __name__ == "__main__":
    main()
