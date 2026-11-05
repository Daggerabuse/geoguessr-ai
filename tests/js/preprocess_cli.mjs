// Runs web/preprocess.js in node for tests/test_web_preprocess.py.
// argv: job.json. Each job: {raw, w, h, channels, tta, out}; raw = uint8 pixels (row-major),
// out receives the float32 (n, 3, S, S) array, plus out + ".resized" with the resized RGB uint8.
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [, , jobPath, modulePath] = process.argv;
const { preprocess, shortSideSize } = await import(pathToFileURL(modulePath).href);
const { cfg, jobs } = JSON.parse(readFileSync(jobPath, "utf8"));
for (const j of jobs) {
  const raw = new Uint8Array(readFileSync(j.raw));
  const r = preprocess(raw, j.w, j.h, j.channels, cfg, j.tta);
  writeFileSync(j.out, Buffer.from(r.data.buffer));
  writeFileSync(j.out + ".resized", Buffer.from(r.resized.buffer));
  j.resized_size = shortSideSize(j.w, j.h, cfg.image_size);
  j.n_crops = r.crops.length;
}
console.log(JSON.stringify(jobs.map((j) => ({ out: j.out, n: j.n_crops, size: j.resized_size }))));
