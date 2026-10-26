import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
// stills: node frames.mjs <html> stills <outdir> t1 t2 ...
// video:  node frames.mjs <html> video <out.mp4> <seconds> [fps]
const [, , file, mode, out, ...rest] = process.argv;
const b = await chromium.launch({ channel: "chrome" });
const p = await b.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1, colorScheme: "light" });
await p.goto("file://" + file);
await p.evaluate(() => window.__ready);
await p.waitForTimeout(800);
const seek = (t) => p.evaluate((t) => window.__seek(t), t);

if (mode === "stills") {
  for (const t of rest) {
    await seek(Number(t));
    await p.waitForTimeout(60);
    await p.screenshot({ path: `${out}/t${String(t).replace(".", "_")}.png`, clip: { x: 0, y: 0, width: 1920, height: 1080 } });
  }
} else {
  const seconds = Number(rest[0]), fps = Number(rest[1] || 30);
  const n = Math.round(seconds * fps);
  const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], { stdio: ["pipe", "inherit", "inherit"] });
  const t0 = Date.now();
  for (let i = 0; i < n; i++) {
    await seek(i / fps);
    const buf = await p.screenshot({ type: "png", clip: { x: 0, y: 0, width: 1920, height: 1080 } });
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once("drain", r));
    if (i % 150 === 0) console.log(`${i}/${n} frames, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await new Promise((r) => ff.on("close", r));
  console.log("done", out);
}
await b.close();
