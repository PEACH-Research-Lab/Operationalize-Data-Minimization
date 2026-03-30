import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const isGitHubActions = process.env.GITHUB_ACTIONS === "true";
const repository = process.env.GITHUB_REPOSITORY || "";
const repoName = repository.split("/")[1] || "";

export default defineConfig({
  base: isGitHubActions && repoName ? `/${repoName}/` : "/",
  plugins: [react()],
});
