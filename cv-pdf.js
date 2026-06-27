const CVPdf = (() => {
  let html2pdfPromise = null;

  function loadHtml2Pdf() {
    if (window.html2pdf) {
      return Promise.resolve(window.html2pdf);
    }
    if (!html2pdfPromise) {
      html2pdfPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js";
        script.async = true;
        script.onload = () => {
          if (window.html2pdf) {
            resolve(window.html2pdf);
            return;
          }
          reject(new Error("html2pdf failed to load"));
        };
        script.onerror = () => reject(new Error("html2pdf script could not be loaded"));
        document.head.appendChild(script);
      });
    }
    return html2pdfPromise;
  }

  function waitForImages(root) {
    const images = Array.from(root.querySelectorAll("img"));
    return Promise.all(
      images.map((img) => {
        if (img.complete && img.naturalWidth > 0) return Promise.resolve();
        return new Promise((resolve) => {
          img.addEventListener("load", resolve, { once: true });
          img.addEventListener("error", resolve, { once: true });
        });
      })
    );
  }

  async function prepareDocument(root, cvReady) {
    if (cvReady) await cvReady;
    if (document.fonts && document.fonts.ready) await document.fonts.ready;
    await waitForImages(root);
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  }

  function hasVisibleContent(element) {
    if (!element) return false;
    const text = element.textContent ? element.textContent.replace(/\s+/g, " ").trim() : "";
    return element.scrollHeight > 40 && text.length > 20;
  }

  function createCaptureClone(element, captureClass) {
    const wrapper = document.createElement("div");
    wrapper.setAttribute("aria-hidden", "true");
    wrapper.style.cssText = [
      "position:fixed",
      "left:-10000px",
      "top:0",
      "width:210mm",
      "background:#fff",
      "z-index:-1",
      "pointer-events:none",
      "overflow:visible",
    ].join(";");
    const clone = element.cloneNode(true);
    clone.classList.add(captureClass);
    wrapper.appendChild(clone);
    document.body.appendChild(wrapper);
    return { clone, wrapper };
  }

  function removeCaptureClone(wrapper) {
    if (wrapper && wrapper.parentNode) {
      wrapper.parentNode.removeChild(wrapper);
    }
  }

  async function setButtonState(button, label, disabled) {
    if (!button) return;
    button.disabled = disabled;
    button.textContent = label;
  }

  async function downloadAsPdf({
    element,
    filename,
    cvReady,
    button,
    captureClass,
    options,
    onProgress,
  }) {
    if (!element) {
      throw new Error("Missing PDF source element");
    }

    const originalLabel = button ? button.textContent : "";
    await setButtonState(button, "Preparing…", true);
    if (onProgress) onProgress("preparing");

    await prepareDocument(element, cvReady);

    if (!hasVisibleContent(element)) {
      throw new Error("CV content is not ready yet. Refresh the page and try again.");
    }

    const { clone, wrapper } = createCaptureClone(element, captureClass);
    try {
      await prepareDocument(clone, null);
      if (!hasVisibleContent(clone)) {
        throw new Error("CV content could not be prepared for export.");
      }

      await setButtonState(button, "Generating PDF…", true);
      if (onProgress) onProgress("generating");

      const html2pdf = await loadHtml2Pdf();
      const captureOptions = {
        margin: options.margin,
        filename,
        image: { type: "jpeg", quality: 0.98 },
        html2canvas: {
          scale: 2,
          useCORS: true,
          backgroundColor: "#ffffff",
          scrollX: 0,
          scrollY: 0,
          windowWidth: clone.scrollWidth,
          logging: false,
        },
        jsPDF: {
          unit: "mm",
          format: "a4",
          orientation: "portrait",
        },
        pagebreak: options.pagebreak,
      };

      await html2pdf().set(captureOptions).from(clone).save();
    } finally {
      removeCaptureClone(wrapper);
      await setButtonState(button, originalLabel, false);
    }
  }

  return {
    downloadAsPdf,
    prepareDocument,
    prefetch: loadHtml2Pdf,
  };
})();
