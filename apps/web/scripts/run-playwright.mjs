import { spawn } from "node:child_process";

const baseUrl = "http://127.0.0.1:3010";
const env = { ...process.env, PLAYWRIGHT_BASE_URL: baseUrl };

async function main() {
  await run("npm.cmd", ["run", "build"]);

  const server = spawn(
    "npm.cmd",
    ["run", "start", "--", "--hostname", "127.0.0.1", "--port", "3010"],
    {
      env,
      stdio: "inherit",
      shell: true
    }
  );

  let closed = false;
  const cleanup = async () => {
    if (closed) {
      return;
    }
    closed = true;
    await terminateProcessTree(server.pid);
  };

  process.on("SIGINT", () => {
    void cleanup().finally(() => process.exit(130));
  });
  process.on("SIGTERM", () => {
    void cleanup().finally(() => process.exit(143));
  });

  try {
    await waitForUrl(`${baseUrl}/login`, 120_000);
    const code = await run("npx.cmd", ["playwright", "test"], { env });
    await cleanup();
    process.exit(code);
  } catch (error) {
    await cleanup();
    throw error;
  }
}

function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: "inherit",
      shell: true,
      ...options
    });

    child.on("exit", (code) => {
      if (code === 0) {
        resolve(0);
        return;
      }
      reject(new Error(`${command} ${args.join(" ")} exited with code ${code ?? -1}`));
    });
    child.on("error", reject);
  });
}

async function waitForUrl(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
    } catch {}
    await delay(1000);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function terminateProcessTree(pid) {
  if (!pid) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    const killer = spawn("taskkill", ["/pid", String(pid), "/t", "/f"], {
      stdio: "ignore",
      shell: true
    });
    killer.on("exit", () => resolve());
    killer.on("error", () => resolve());
  });
}

void main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
