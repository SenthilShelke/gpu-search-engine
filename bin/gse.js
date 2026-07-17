#!/usr/bin/env node
// CLI entrypoint for gpu-search-engine.
//
// Responsibilities:
//   1. Embed the user's query text in Node (no Python at runtime) using
//      @huggingface/transformers running the same all-MiniLM-L6-v2 model the
//      corpus was originally embedded with.
//   2. Spawn the prebuilt Metal search binary and speak its existing
//      stdin/stdout protocol: write a 384 x float32 (little-endian) query
//      vector in, read back 5 x int32 (little-endian) result indices.
//   3. Look up display text for those indices from the bundled corpus.json
//      and print results.
//
// The GPU work (the actual point of this project) happens entirely inside
// the native binary via a custom Metal compute kernel (src/compute.metal).
// This file is intentionally just plumbing around that.

import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";
import { spawn } from "node:child_process";
import readline from "node:readline";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = path.resolve(__dirname, "..");
const NATIVE_DIR = path.join(PACKAGE_ROOT, "native");
const BINARY_PATH = path.join(NATIVE_DIR, "main-darwin-arm64");
const CORPUS_PATH = path.join(NATIVE_DIR, "data", "corpus.json");

const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2"; // model repo id (unchanged; @huggingface/transformers still resolves Xenova/* ONNX exports)
const VECTOR_DIM = 384;
const TOP_K = 5;

function loadCorpus() {
  if (!fs.existsSync(CORPUS_PATH)) {
    console.error(
      `Missing corpus data at ${CORPUS_PATH}. The package may be corrupted; try reinstalling.`
    );
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(CORPUS_PATH, "utf8"));
}

function spawnEngine() {
  if (!fs.existsSync(BINARY_PATH)) {
    console.error(
      `Missing native search binary at ${BINARY_PATH}. The package may be corrupted; try reinstalling.`
    );
    process.exit(1);
  }
  // cwd matters here: the binary opens "data/vector_db.bin" as a path
  // relative to its working directory, and Metal's newDefaultLibrary looks
  // for default.metallib alongside the executable/working directory. Both
  // are satisfied by running from native/.
  const engine = spawn(BINARY_PATH, [], { cwd: NATIVE_DIR });
  engine.on("error", (err) => {
    console.error(`Failed to start the search engine binary: ${err.message}`);
    process.exit(1);
  });
  // The engine can exit almost immediately after spawning (e.g. if it can't
  // open native/data/vector_db.bin) -- often before the embedding model has
  // even finished loading, well before search() gets a chance to attach its
  // own listeners. Node's "exit" event doesn't replay for late listeners, so
  // record exit state here, from the moment the process is spawned, and let
  // search() consult it instead of relying on catching the event live.
  engine.exitInfo = null;
  engine.on("exit", (code, signal) => {
    engine.exitInfo = { code, signal };
  });
  return engine;
}

async function loadEmbedder() {
  const { pipeline } = await import("@huggingface/transformers");
  return pipeline("feature-extraction", EMBEDDING_MODEL);
}

async function embedQuery(embedder, text) {
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Float32Array.from(output.data);
}

function search(engine, vector) {
  return new Promise((resolve, reject) => {
    // The engine may have already exited (e.g. failed to open
    // data/vector_db.bin at startup) before this search() call even
    // started -- check the exit state recorded by spawnEngine() up front,
    // since the "exit" event itself won't replay for a listener added now.
    if (engine.exitInfo) {
      reject(
        new Error(
          `Search engine exited unexpectedly (code ${engine.exitInfo.code}, signal ${engine.exitInfo.signal}) before returning results.`
        )
      );
      return;
    }

    const resultBytes = TOP_K * 4; // 5 x int32
    let received = Buffer.alloc(0);
    let settled = false;

    function cleanup() {
      settled = true;
      engine.stdout.removeListener("data", onData);
      engine.removeListener("exit", onExit);
      engine.stdin.removeListener("error", onStdinError);
    }

    function onData(chunk) {
      received = Buffer.concat([received, chunk]);
      if (received.length >= resultBytes) {
        cleanup();
        const indices = [];
        for (let i = 0; i < TOP_K; i++) {
          indices.push(received.readInt32LE(i * 4));
        }
        resolve(indices);
      }
    }

    // The engine can exit before (or instead of) returning a result, e.g. if
    // it can't open data/vector_db.bin at startup. Without this, the promise
    // would otherwise hang forever waiting for stdout bytes that will never
    // arrive.
    function onExit(code, signal) {
      if (settled) return;
      cleanup();
      reject(
        new Error(
          `Search engine exited unexpectedly (code ${code}, signal ${signal}) before returning results.`
        )
      );
    }

    // Writing to the engine's stdin after it has already exited (e.g. it
    // failed to start up) emits an error on the stdin stream itself, not on
    // `engine` -- listen for it explicitly so it doesn't go uncaught.
    function onStdinError(err) {
      if (settled) return;
      cleanup();
      reject(err);
    }

    engine.stdout.on("data", onData);
    engine.once("exit", onExit);
    engine.once("error", (err) => {
      if (settled) return;
      cleanup();
      reject(err);
    });
    engine.stdin.once("error", onStdinError);

    const buf = Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength);
    engine.stdin.write(buf);
  });
}

function printResults(indices, corpus) {
  console.log();
  for (const index of indices) {
    console.log(corpus[index] ?? `<missing corpus entry at index ${index}>`);
  }
  console.log();
}

// Ends the engine subprocess's stdin, letting its `while(cin.read(...))` loop
// exit and main() return normally. Do NOT SIGTERM/kill() it directly --
// interrupting Metal's command queue/runtime mid-teardown with a signal
// reliably crashes it (libc++abi mutex lock failure). If it doesn't exit on
// its own within a few seconds (shouldn't happen), fall back to killing it
// so the CLI doesn't hang forever.
function shutdownEngine(engine) {
  return new Promise((resolve) => {
    const forceKillTimer = setTimeout(() => {
      engine.kill();
      resolve();
    }, 3000);
    engine.once("exit", () => {
      clearTimeout(forceKillTimer);
      resolve();
    });
    engine.stdin.end();
  });
}

// IMPORTANT: never call process.exit() anywhere in this file once the
// embedding model has been loaded. @huggingface/transformers' ONNX runtime
// keeps native worker threads alive; forcibly exiting the process (rather
// than letting it exit naturally once all handles/timers close) races their
// teardown and reliably crashes with a libc++abi "mutex lock failed" abort.
// Once shutdownEngine() resolves and readline has closed, Node will exit on
// its own as soon as the event loop has nothing left to keep it alive.

async function runOneShot(query) {
  const corpus = loadCorpus();
  const engine = spawnEngine();
  // On first run, @huggingface/transformers downloads the ONNX model (~90MB)
  // to its own cache inside node_modules; this can take a while on a slow
  // connection, so surface a status line rather than hanging silently.
  console.error("Loading embedding model...");
  const embedder = await loadEmbedder();

  const vector = await embedQuery(embedder, query);
  const start = performance.now();
  const indices = await search(engine, vector);
  const elapsedMs = performance.now() - start;

  console.log(`\nLatency: ${elapsedMs.toFixed(2)} ms`);
  printResults(indices, corpus);

  await shutdownEngine(engine);
}

async function runRepl() {
  const corpus = loadCorpus();
  const engine = spawnEngine();
  console.log("Loading embedding model...");
  const embedder = await loadEmbedder();
  console.log("Ready. Type a query and press enter (Ctrl+C to exit).\n");

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  rl.setPrompt("Search: ");
  rl.prompt();

  // Tracks the in-flight/queued search chain so "close" (stdin EOF / Ctrl+D)
  // can wait for it before tearing down the engine subprocess, and so
  // back-to-back queries (e.g. multiple lines arriving from piped stdin
  // before the first search resolves) are sent to the engine one at a time.
  // The engine binary only ever has one query in flight at once -- sending a
  // second query before reading the first's result would desync the
  // stdin/stdout protocol and cause later queries to read stale/wrong bytes.
  let pendingSearch = Promise.resolve();
  let closed = false;

  rl.on("line", (line) => {
    const query = line.trim();
    if (query.length === 0) {
      if (!closed) rl.prompt();
      return;
    }
    pendingSearch = pendingSearch.then(async () => {
      try {
        const vector = await embedQuery(embedder, query);
        const start = performance.now();
        const indices = await search(engine, vector);
        const elapsedMs = performance.now() - start;
        console.log(`\nLatency: ${elapsedMs.toFixed(2)} ms`);
        printResults(indices, corpus);
      } catch (err) {
        console.error(`Search failed: ${err.message}`);
      }
      // stdin may have closed (EOF) while this search was in flight; the
      // readline interface is unusable at that point, so skip re-prompting.
      if (!closed) rl.prompt();
    });
  });

  rl.on("close", async () => {
    closed = true;
    await pendingSearch;
    await shutdownEngine(engine);
    // Deliberately no process.exit() here -- see comment above shutdownEngine.
    // Node exits naturally once the engine subprocess handle and readline's
    // stdin listener are both gone, with no exception risk from the ONNX
    // runtime's native threads.
  });
}

async function main() {
  const args = process.argv.slice(2);
  const query = args.join(" ").trim();
  if (args.length === 0) {
    await runRepl();
  } else if (query.length === 0) {
    console.error("Search query is empty. Usage: gse \"<query>\"");
    process.exitCode = 1;
  } else {
    await runOneShot(query);
  }
}

main().catch((err) => {
  console.error(err.message ?? err);
  // Use process.exitCode (lets Node exit naturally once pending work
  // finishes) rather than process.exit(1) -- see the comment above
  // shutdownEngine() for why calling process.exit() after the embedding
  // model has loaded crashes the process.
  process.exitCode = 1;
});
