import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#090b0f",
        panel: "#11151c",
        panelSoft: "#171d26",
        border: "#27303d",
        accent: "#36c2a0",
        warning: "#f6b44b",
        danger: "#ef6262"
      }
    }
  },
  plugins: []
};

export default config;

