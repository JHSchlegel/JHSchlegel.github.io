<script>
/* ============================================================
   Dark / light mode icon for Quarto's color-scheme toggle.
   Quarto ships the toggle with an empty <i class="bi">; this adds
   a Bootstrap-Icons glyph and swaps it whenever the scheme changes:
       light mode  ->  bi-moon-stars   (click to go dark)
       dark  mode  ->  bi-sun          (click to go light)
   Quarto marks the toggle with the class `alternate` while dark.
   ============================================================ */
(function () {
  function apply() {
    document.querySelectorAll(".quarto-color-scheme-toggle").forEach(function (toggle) {
      var icon = toggle.querySelector(".bi");
      if (!icon) return;
      var dark = toggle.classList.contains("alternate");
      icon.classList.remove("bi-sun", "bi-moon-stars");
      icon.classList.add(dark ? "bi-sun" : "bi-moon-stars");
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    apply();
    document.querySelectorAll(".quarto-color-scheme-toggle").forEach(function (toggle) {
      new MutationObserver(apply).observe(toggle, {
        attributes: true,
        attributeFilter: ["class"]
      });
    });
  });
})();
</script>
