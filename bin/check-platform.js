#!/usr/bin/env node
// Runs as a postinstall step. package.json's "os"/"cpu" fields already block
// npm from installing on unsupported platforms, but npm's enforcement of
// those fields is a warning in some configurations (e.g. --force), so this
// gives a clear, explicit failure instead of a confusing runtime crash later.

if (process.platform !== "darwin") {
  console.error(
    `gpu-search-engine requires macOS (Metal GPU support). Detected platform: ${process.platform}.`
  );
  process.exit(1);
}

if (process.arch !== "arm64") {
  console.error(
    `gpu-search-engine ships a prebuilt Apple Silicon (arm64) binary. Detected arch: ${process.arch}. ` +
      `Intel Macs are not currently supported.`
  );
  process.exit(1);
}
