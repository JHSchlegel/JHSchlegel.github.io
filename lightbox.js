<script>
/* ============================================================
   Lightbox — full-screen image viewer with prev/next navigation.
   Targets content images (photography albums + blog figures/plots)
   inside the main document, excluding the portrait, navbar logo,
   listing thumbnails, and images that are already links.
   Wired site-wide via include-after-body in _quarto.yml.
   ============================================================ */
(function () {
  document.addEventListener("DOMContentLoaded", function () {
    function eligible(img) {
      if (!img.closest("#quarto-document-content")) return false;
      if (img.closest(".jhs-hero")) return false;
      if (img.closest(".navbar")) return false;
      if (img.closest(".quarto-listing")) return false;
      if (img.closest("a")) return false;
      return true;
    }

    var items = Array.prototype.filter.call(
      document.querySelectorAll("#quarto-document-content img"), eligible
    );
    if (!items.length) return;

    var current = 0;
    var lastFocus = null;

    // ---- Build the overlay once ----
    var lb = document.createElement("div");
    lb.className = "jhs-lb";
    lb.setAttribute("role", "dialog");
    lb.setAttribute("aria-modal", "true");
    lb.setAttribute("aria-label", "Image viewer");
    lb.hidden = true;
    lb.innerHTML =
      '<button class="jhs-lb-close" aria-label="Close (Esc)"><i class="bi bi-x-lg"></i></button>' +
      '<button class="jhs-lb-prev" aria-label="Previous image"><i class="bi bi-chevron-left"></i></button>' +
      '<figure class="jhs-lb-figure"><img class="jhs-lb-img" alt=""><figcaption class="jhs-lb-caption"></figcaption></figure>' +
      '<button class="jhs-lb-next" aria-label="Next image"><i class="bi bi-chevron-right"></i></button>' +
      '<div class="jhs-lb-counter"></div>';
    document.body.appendChild(lb);

    var imgEl = lb.querySelector(".jhs-lb-img");
    var capEl = lb.querySelector(".jhs-lb-caption");
    var counterEl = lb.querySelector(".jhs-lb-counter");
    var prevBtn = lb.querySelector(".jhs-lb-prev");
    var nextBtn = lb.querySelector(".jhs-lb-next");
    var closeBtn = lb.querySelector(".jhs-lb-close");
    var figEl = lb.querySelector(".jhs-lb-figure");

    var single = items.length < 2;
    prevBtn.style.display = single ? "none" : "";
    nextBtn.style.display = single ? "none" : "";
    counterEl.style.display = single ? "none" : "";

    function show(i) {
      current = (i + items.length) % items.length;
      var img = items[current];
      imgEl.src = img.currentSrc || img.src;
      var alt = img.getAttribute("alt") || "";
      imgEl.alt = alt;
      capEl.textContent = alt;
      capEl.style.display = alt ? "" : "none";
      counterEl.textContent = (current + 1) + " / " + items.length;
    }
    function open(i) {
      lastFocus = document.activeElement;
      show(i);
      lb.hidden = false;
      document.body.classList.add("jhs-lb-open");
      closeBtn.focus();
    }
    function close() {
      lb.hidden = true;
      document.body.classList.remove("jhs-lb-open");
      imgEl.removeAttribute("src");
      if (lastFocus && lastFocus.focus) lastFocus.focus();
    }
    function next() { show(current + 1); }
    function prev() { show(current - 1); }

    // ---- Make eligible images clickable ----
    items.forEach(function (img, i) {
      img.classList.add("jhs-lb-eligible");
      img.addEventListener("click", function () { open(i); });
    });

    // ---- Controls ----
    closeBtn.addEventListener("click", close);
    nextBtn.addEventListener("click", function (e) { e.stopPropagation(); next(); });
    prevBtn.addEventListener("click", function (e) { e.stopPropagation(); prev(); });
    lb.addEventListener("click", function (e) {
      if (e.target === lb || e.target === figEl) close();
    });
    document.addEventListener("keydown", function (e) {
      if (lb.hidden) return;
      if (e.key === "Escape") close();
      else if (e.key === "ArrowRight") next();
      else if (e.key === "ArrowLeft") prev();
    });
  });
})();
</script>
