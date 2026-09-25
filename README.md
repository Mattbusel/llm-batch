# llm-batch

[![CI](https://github.com/Mattbusel/llm-batch/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/llm-batch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Single header](https://img.shields.io/badge/single-header-green.svg)

Run thousands of LLM prompts from C++ with a thread pool, rate limiting and crash-safe checkpoints.

> Part of **[llm-cpp](https://github.com/Mattbusel/llm-cpp)**, a family of 26 single-header C++ libraries for building on LLM APIs. Each one stands alone: copy one header, include it, done.

Pushing a large prompt list through an API one call at a time is slow, and a crash halfway through means paying for the same calls twice. llm-batch runs prompts in parallel under a requests-per-second cap and writes a checkpoint as each one finishes, so a rerun skips what is already done.

## Features

- Worker thread pool (`num_threads`, default 4)
- Global rate limit in requests per second (`rate_limit_rps`, default 5; 0 disables)
- Checkpointing: completed item ids are appended to `checkpoint_path`, and a rerun skips them
- JSONL in, JSONL out: `process_file()` reads `{"id", "prompt", "system"}` lines and writes one result per line
- Per-item system prompt, latency measurement, success flag and error message
- Thread-safe progress callback with done/total counts

## Quick start

Requirements: a C++17 compiler and libcurl (`apt install libcurl4-openssl-dev`, `brew install curl`, or `vcpkg install curl`).

1. Copy [`include/llm_batch.hpp`](include/llm_batch.hpp) into your project.
2. In exactly one `.cpp` file, `#define LLM_BATCH_IMPLEMENTATION` before including it. Other files just `#include "llm_batch.hpp"`.

```cpp
#define LLM_BATCH_IMPLEMENTATION
#include "llm_batch.hpp"
#include <cstdlib>
#include <iostream>

int main() {
    llm::BatchConfig cfg;
    cfg.api_key         = std::getenv("OPENAI_API_KEY");
    cfg.num_threads     = 4;
    cfg.rate_limit_rps  = 5.0;
    cfg.checkpoint_path = "batch.checkpoint";  // rerun to skip items already done
    cfg.on_progress = [](const llm::BatchResult& r, size_t done, size_t total) {
        std::cout << "[" << done << "/" << total << "] " << r.id << "\n";
    };

    std::vector<llm::BatchItem> items = {
        {"q1", "Summarize the plot of Hamlet in one sentence.", ""},
        {"q2", "Name three prime numbers above 100.", ""},
        {"q3", "Translate 'good morning' into Spanish.", ""},
    };

    for (const auto& r : llm::process_batch(items, cfg))
        std::cout << r.id << ": " << (r.success ? r.response : "ERROR " + r.error) << "\n";
}
```

Build and run:

```bash
g++ -std=c++17 -I include example.cpp -o example -lcurl -pthread
export OPENAI_API_KEY=sk-...
./example
```

## API

Everything lives in namespace `llm`.

| Function / type | What it does |
|---|---|
| `process_batch(items, cfg)` | Process an in-memory `std::vector<BatchItem>` and return one `BatchResult` per item |
| `process_file(input_path, output_path, cfg)` | Read prompts from a JSONL file, write results as JSONL, return the success count |
| `BatchConfig` | API key, model, endpoint URL, max tokens, temperature, threads, rate limit, checkpoint path, verbose, `on_progress` |

## How it works

Items go into a shared queue served by `num_threads` workers. Before each request a worker waits on a shared rate limiter, then POSTs to the chat completions endpoint through libcurl. When a result comes back it is appended to the checkpoint file and passed to `on_progress`. On startup, ids already in the checkpoint are marked `[skipped - checkpoint]` instead of being sent again.

## Examples

The [`examples/`](examples) folder has runnable programs:

- [`basic_batch.cpp`](examples/basic_batch.cpp)
- [`checkpoint_batch.cpp`](examples/checkpoint_batch.cpp)
- [`file_batch.cpp`](examples/file_batch.cpp)
- [`resumable_batch.cpp`](examples/resumable_batch.cpp)

Build the examples with CMake (needs libcurl):

```bash
cmake -B build
cmake --build build
```

Examples that call the API read `OPENAI_API_KEY` from the environment.

## Limitations

- Uses the synchronous chat completions endpoint, not OpenAI's asynchronous Batch API.
- Failed requests are reported, not retried.
- Checkpoints record ids only; responses for skipped items are not reloaded from a previous run.

## License

MIT. See [LICENSE](LICENSE).
