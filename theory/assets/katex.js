// Renders the spans and divs that pymdownx.arithmatex leaves behind.
// `document$` is Material's page-load stream, so this also runs after instant navigation.
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
  });
});
