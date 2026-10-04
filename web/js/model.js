// MLP exportado por export_web_model.py (Linear -> ReLU -> ... -> Linear -> softmax).

export async function loadModel(url) {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new GestureModel(await res.json());
}

class GestureModel {
  constructor({ classes, layers }) {
    this.classes = classes;
    this.layers = layers;
  }

  /** x: array plano de 63 valores. Devuelve { index, conf, probs }. */
  predict(x) {
    let h = x;
    this.layers.forEach(({ W, b }, i) => {
      const out = b.map((bias, j) => {
        let s = bias;
        const row = W[j];
        for (let k = 0; k < row.length; k++) s += row[k] * h[k];
        return s;
      });
      h = i < this.layers.length - 1 ? out.map((v) => Math.max(0, v)) : out;
    });
    const max = Math.max(...h);
    const exp = h.map((v) => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    const probs = exp.map((v) => v / sum);
    const index = probs.indexOf(Math.max(...probs));
    return { index, conf: probs[index], probs };
  }
}
