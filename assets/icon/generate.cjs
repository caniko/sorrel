const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { readFileSync, writeFileSync } = require("node:fs");
const { resolve } = require("node:path");

const root = resolve(__dirname, "../..");
const source = resolve(root, "logo/logo-clean.svg");
const check = process.argv[2] === "--check";
assert(process.argv.length === (check ? 3 : 2), "Usage: node assets/icon/generate.cjs [--check]");
const save = (path, bytes) => {
  const target = resolve(root, path);
  if (check) assert(readFileSync(target).equals(bytes), `${path} is out of date`);
  else writeFileSync(target, bytes);
};

save("assets/icon/sorrel.svg", readFileSync(source));
const sizes = [16, 32, 48, 64, 128, 256, 512];
const pngs = sizes.map((size) => {
  const png = execFileSync("resvg", ["-w", `${size}`, "-h", `${size}`, source, "-c"]);
  save(`assets/icon/sorrel-${size}.png`, png);
  return png;
});

// PNG-compressed ICO entries keep both the source asset and packaged crate small.
const entries = pngs.slice(0, -1);
const header = Buffer.alloc(6 + entries.length * 16);
header.writeUInt16LE(1, 2);
header.writeUInt16LE(entries.length, 4);
let offset = header.length;
entries.forEach((png, index) => {
  const entry = 6 + index * 16;
  header[entry] = header[entry + 1] = sizes[index] % 256;
  header.writeUInt16LE(1, entry + 4);
  header.writeUInt16LE(32, entry + 6);
  header.writeUInt32LE(png.length, entry + 8);
  header.writeUInt32LE(offset, entry + 12);
  offset += png.length;
});
const ico = Buffer.concat([header, ...entries]);
save("assets/icon/sorrel.ico", ico);
save("crates/sorrel/assets/sorrel.ico", ico);
save("crates/sorrel/assets/sorrel-256.png", pngs[sizes.indexOf(256)]);
console.log(check ? "Logo assets are up to date." : "Logo assets regenerated.");
