// Unlimited-OCR-2 cookbook config. Consumed by _deployment.jsx + _playground.jsx.

export const config = {
  modelName: "Unlimited-OCR-2",

  supportedHardware: ["l20", "mac"],

  // L20 is not in the shared HARDWARE_CATALOG.
  hardware: [
    { id: "l20", label: "L20", vram: "48GB", vendor: "nvidia" },
    // Apple Silicon Mac (MLX backend, unified memory). Verified on an M5 16GB.
    { id: "mac", label: "Apple Silicon", vram: "16GB+", vendor: "apple" },
  ],

  variants: [{ id: "default", label: "Default" }],
  quantizations: [
    { id: "bf16", label: "BF16" },
    // mlx-community exports (mlx-vlm layout); Apple Silicon only.
    { id: "mlx-8bit", label: "MLX 8-bit" },
    { id: "mlx-6bit", label: "MLX 6-bit" },
    { id: "mlx-4bit", label: "MLX 4-bit" },
  ],
  // No Docker path on Apple Silicon: the MLX backend runs native-only.
  runModes: (s) => (s.hw === "mac" ? ["python"] : ["python", "docker"]),
  strategies: [{ id: "balanced", label: "Balanced" }],
  nodesOptions: [{ id: "single", label: "Single Node" }],

  modelNames: {
    "default|bf16": "baidu/Unlimited-OCR-2",
    "default|mlx-8bit": "mlx-community/Unlimited-OCR-2-8bit",
    "default|mlx-6bit": "mlx-community/Unlimited-OCR-2-6bit",
    "default|mlx-4bit": "mlx-community/Unlimited-OCR-2-4bit",
  },

  placeholders: {
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    HF_TOKEN: {
      target: "command",
      label: "HF token (Docker)",
      default: "<your-hf-token>",
    },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{
  "model": "{{MODEL_NAME}}",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "https://example.com/page_1.png"}},
      {"type": "image_url", "image_url": {"url": "https://example.com/page_2.png"}},
      {"type": "text", "text": "<PAGE><image><PAGE><image>Multi page parsing."}
    ]
  }],
  "images_config": {"image_mode": "base"},
  "skip_special_tokens": false,
  "max_tokens": 16000
}'`,

  dockerImages: {
    l20: "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "baidu/Unlimited-OCR-2",
  },

  // Speculative decoding, PD disaggregation and pipeline parallelism are
  // rejected at launch; the radix cache is always off, so HiCache has no use.
  playgroundFeatures: {
    attention: {
      knobs: [
        // The MLX backend serves on one device.
        { id: "tp", label: "TP", values: [null, 1, { value: 2, disable: { hw: ["mac"] } }] },
      ],
    },
  },

  cells: [
    {
      match: {
        hw: "l20",
        variant: "default",
        quant: "bf16",
        strategy: "balanced",
        nodes: "single",
      },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--chat-template unlimited-ocr",
        "--uocr-doc-split",
        "--enable-streaming-session",
        "--uocr-max-page-tokens 8000",
        "--cuda-graph-backend-decode full",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "mac",
        variant: "default",
        quant: "bf16",
        strategy: "balanced",
        nodes: "single",
      },
      verified: true,
      env: ["SGLANG_USE_MLX=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--chat-template unlimited-ocr",
        "--uocr-doc-split",
        "--enable-streaming-session",
        "--uocr-max-page-tokens 8000",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "mac",
        variant: "default",
        quant: "mlx-8bit",
        strategy: "balanced",
        nodes: "single",
      },
      verified: true,
      env: ["SGLANG_USE_MLX=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--chat-template unlimited-ocr",
        "--uocr-doc-split",
        "--enable-streaming-session",
        "--uocr-max-page-tokens 8000",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "mac",
        variant: "default",
        quant: "mlx-6bit",
        strategy: "balanced",
        nodes: "single",
      },
      verified: true,
      env: ["SGLANG_USE_MLX=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--chat-template unlimited-ocr",
        "--uocr-doc-split",
        "--enable-streaming-session",
        "--uocr-max-page-tokens 8000",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: {
        hw: "mac",
        variant: "default",
        quant: "mlx-4bit",
        strategy: "balanced",
        nodes: "single",
      },
      verified: true,
      env: ["SGLANG_USE_MLX=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--chat-template unlimited-ocr",
        "--uocr-doc-split",
        "--enable-streaming-session",
        "--uocr-max-page-tokens 8000",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
