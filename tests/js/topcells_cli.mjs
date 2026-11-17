// Runs topCells of web/preprocess.js in node for tests/test_web_preprocess.py.
// argv: job.json {emb, head (flat), scale, k}, module path. Prints [[cell, prob], ...].
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [, , jobPath, modulePath] = process.argv;
const { topCells } = await import(pathToFileURL(modulePath).href);
const { emb, head, scale, k } = JSON.parse(readFileSync(jobPath, "utf8"));
console.log(JSON.stringify(topCells(Float32Array.from(emb), Float32Array.from(head), scale, k)));
