// Tailwind was evaluated and dropped: every surface in this console is styled by
// hand in src/index.css, so the framework's preflight was the only part in use.
// Autoprefixer stays for vendor prefixes on backdrop-filter and friends.
export default {
  plugins: {
    autoprefixer: {},
  },
}
