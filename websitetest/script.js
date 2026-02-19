document.getElementById("year").textContent = new Date().getFullYear();

async function loadText(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path} (HTTP ${res.status})`);
  return await res.text();
}

function showError(el, err) {
  if (!el) return;
  el.textContent =
    `Could not load file.\n\n` +
    `${err.message}\n\n` +
    `Fix:\n` +
    `• Check the file exists in your project folder.\n` +
    `• Check spelling/case matches exactly.\n`;
}

// ---- Project 1 (Transformer) ----
const P1_CODE = "assets/code/EEEE3129_Coursework1_extracted.py";
const P1_DATA = "assets/data/factoryReports_export.csv";

// ---- Project 2 (Forecasting) ----
const P2_CODE = "assets/code/EEEE3129_Coursework2_extracted.py";
const P2_DATA = "assets/data/UKLoad2023_preview.json";

// ---- Project 3 (YOLO config viewer) ----
const YOLO_ARGS = "assets/yolo/yolo_args.yaml";

(async () => {
  // P1 code
  try {
    const code = await loadText(P1_CODE);
    document.getElementById("codePathLabelP1").textContent = P1_CODE;
    document.getElementById("codeViewerP1").textContent = code;
  } catch (err) {
    showError(document.getElementById("codeViewerP1"), err);
  }

  // P1 dataset preview (first 25 rows)
  try {
    const csv = await loadText(P1_DATA);
    const lines = csv.split(/\r?\n/).slice(0, 26).join("\n");
    document.getElementById("dataPathLabelP1").textContent = P1_DATA;
    document.getElementById("dataViewerP1").textContent = lines;
  } catch (err) {
    showError(document.getElementById("dataViewerP1"), err);
  }

  // P2 code
  try {
    const code = await loadText(P2_CODE);
    document.getElementById("codePathLabelP2").textContent = P2_CODE;
    document.getElementById("codeViewerP2").textContent = code;
  } catch (err) {
    showError(document.getElementById("codeViewerP2"), err);
  }

  // P2 dataset preview (JSON text)
  try {
    const preview = await loadText(P2_DATA);
    document.getElementById("dataPathLabelP2").textContent = P2_DATA;
    document.getElementById("dataViewerP2").textContent = preview;
  } catch (err) {
    showError(document.getElementById("dataViewerP2"), err);
  }

  // YOLO args.yaml viewer
  try {
    const yml = await loadText(YOLO_ARGS);
    document.getElementById("yoloArgsPathLabel").textContent = YOLO_ARGS;
    document.getElementById("yoloArgsViewer").textContent = yml;
  } catch (err) {
    showError(document.getElementById("yoloArgsViewer"), err);
  }
})();

// ===== Lightbox for zoomable images =====
(() => {
  const lb = document.getElementById("lightbox");
  const lbImg = document.getElementById("lightboxImg");
  const lbCap = document.getElementById("lightboxCap");

  if (!lb || !lbImg || !lbCap) return;

  function openLightbox(src, alt, cap) {
    lb.classList.add("open");
    lb.setAttribute("aria-hidden", "false");
    lbImg.src = src;
    lbImg.alt = alt || "";
    lbCap.textContent = cap || "";
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    lb.classList.remove("open");
    lb.setAttribute("aria-hidden", "true");
    lbImg.src = "";
    lbImg.alt = "";
    lbCap.textContent = "";
    document.body.style.overflow = "";
  }

  document.addEventListener("click", (e) => {
    const img = e.target.closest("img.zoomable");
    if (img) {
      openLightbox(img.src, img.alt, img.dataset.caption);
      return;
    }
    if (e.target && e.target.dataset && e.target.dataset.close === "true") {
      closeLightbox();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLightbox();
  });
})();
