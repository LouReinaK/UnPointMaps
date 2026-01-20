import js from "@eslint/js";

export default [
  js.configs.recommended,
  {
    languageOptions: {
      ecmaVersion: 2021,
      sourceType: "module",
      globals: {
        document: "readonly",
        window: "readonly",
        L: "readonly", // Leaflet
        console: "readonly",
        WebSocket: "readonly",
        fetch: "readonly",
        alert: "readonly",
        setTimeout: "readonly"
      }
    },
    rules: {
      indent: ["error", 2],
      "linebreak-style": ["error", "windows"],
      quotes: ["error", "single"],
      semi: ["error", "always"]
    }
  },
  {
    files: ["static/app.js"]
  }
];