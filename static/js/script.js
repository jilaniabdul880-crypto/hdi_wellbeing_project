(() => {
  "use strict";

  // ---- Constants mirroring the server-side UNDP formula (for live hero preview) ----
  const LE_MIN = 20, LE_MAX = 85;
  const MYS_MAX = 15;
  const EYS_MAX = 18;
  const GNI_MIN = 100, GNI_MAX = 75000;

  const TIER_COLORS = {
    "Low": "#E76F51",
    "Medium": "#F4A261",
    "High": "#2A9D8F",
    "Very High": "#264653",
  };

  const fields = ["life_expectancy", "mean_years_schooling", "expected_years_schooling", "gni_per_capita"];

  const sliders = {};
  fields.forEach((f) => { sliders[f] = document.getElementById(f); });

  const valueOutputs = {
    life_expectancy: document.getElementById("life_expectancy_val"),
    mean_years_schooling: document.getElementById("mean_years_schooling_val"),
    expected_years_schooling: document.getElementById("expected_years_schooling_val"),
    gni_per_capita: document.getElementById("gni_per_capita_val"),
  };

  const heroMarker = document.getElementById("heroMarker");
  const resultMarker = document.getElementById("resultMarker");
  const resultEmpty = document.getElementById("resultEmpty");
  const resultContent = document.getElementById("resultContent");
  const resultError = document.getElementById("resultError");
  const form = document.getElementById("predictForm");

  function fmtGni(v) {
    return "$" + Math.round(v).toLocaleString("en-US");
  }

  function refreshLabels() {
    valueOutputs.life_expectancy.textContent = parseFloat(sliders.life_expectancy.value).toFixed(1);
    valueOutputs.mean_years_schooling.textContent = parseFloat(sliders.mean_years_schooling.value).toFixed(1);
    valueOutputs.expected_years_schooling.textContent = parseFloat(sliders.expected_years_schooling.value).toFixed(1);
    valueOutputs.gni_per_capita.textContent = fmtGni(sliders.gni_per_capita.value);
  }

  function quickHdi() {
    const le = parseFloat(sliders.life_expectancy.value);
    const mys = parseFloat(sliders.mean_years_schooling.value);
    const eys = parseFloat(sliders.expected_years_schooling.value);
    const gni = parseFloat(sliders.gni_per_capita.value);

    const lei = Math.max((le - LE_MIN) / (LE_MAX - LE_MIN), 1e-6);
    const mysi = Math.min(mys / MYS_MAX, 1);
    const eysi = Math.min(eys / EYS_MAX, 1);
    const ei = Math.max((mysi + eysi) / 2, 1e-6);
    const gniClamped = Math.min(Math.max(gni, GNI_MIN), GNI_MAX);
    const ii = Math.max((Math.log(gniClamped) - Math.log(GNI_MIN)) / (Math.log(GNI_MAX) - Math.log(GNI_MIN)), 1e-6);

    return Math.cbrt(lei * ei * ii);
  }

  function updateHeroMarker() {
    const score = quickHdi();
    heroMarker.style.left = (score * 100).toFixed(1) + "%";
  }

  function updateResultMarker(score) {
    resultMarker.style.left = (score * 100).toFixed(1) + "%";
  }

  fields.forEach((f) => {
    sliders[f].addEventListener("input", () => {
      refreshLabels();
      updateHeroMarker();
    });
  });

  refreshLabels();
  updateHeroMarker();

  // ---- Presets ----
  const PRESETS = {
    veryhigh: { life_expectancy: 82, mean_years_schooling: 13.2, expected_years_schooling: 17, gni_per_capita: 52000 },
    medium:   { life_expectancy: 68, mean_years_schooling: 7.5, expected_years_schooling: 11.5, gni_per_capita: 8500 },
    low:      { life_expectancy: 55, mean_years_schooling: 3.2, expected_years_schooling: 7, gni_per_capita: 1400 },
  };

  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = PRESETS[btn.dataset.preset];
      fields.forEach((f) => { sliders[f].value = preset[f]; });
      refreshLabels();
      updateHeroMarker();
      form.requestSubmit();
    });
  });

  // ---- Submit / predict ----
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const payload = {};
    fields.forEach((f) => { payload[f] = parseFloat(sliders[f].value); });

    const btn = form.querySelector(".btn-predict span");
    const originalLabel = btn.textContent;
    btn.textContent = "Estimating…";

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok) {
        showError(data.error || "Something went wrong.");
        return;
      }
      renderResult(data);
    } catch (err) {
      showError("Could not reach the prediction service. Please try again.");
    } finally {
      btn.textContent = originalLabel;
    }
  });

  function showError(message) {
    resultEmpty.classList.add("hidden");
    resultContent.classList.add("hidden");
    resultError.classList.remove("hidden");
    resultError.textContent = message;
  }

  function renderResult(data) {
    resultError.classList.add("hidden");
    resultEmpty.classList.add("hidden");
    resultContent.classList.remove("hidden");

    document.getElementById("resultTierName").textContent = data.predicted_tier;
    document.getElementById("resultTierName").style.color = TIER_COLORS[data.predicted_tier] || "inherit";
    document.getElementById("resultBlurb").textContent = data.tier_info.blurb;
    document.getElementById("scoreNumber").textContent = data.hdi_score.toFixed(3);
    document.getElementById("formulaTier").textContent = data.formula_tier;

    updateResultMarker(data.hdi_score);

    const probsList = document.getElementById("probsList");
    probsList.innerHTML = "";
    data.tier_order.forEach((tier) => {
      const pct = (data.probabilities[tier] || 0) * 100;
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span class="prob-name">${tier}</span>
        <span class="prob-track"><span class="prob-fill" style="width:${pct.toFixed(1)}%; background:${TIER_COLORS[tier]}"></span></span>
        <span class="prob-pct">${pct.toFixed(1)}%</span>
      `;
      probsList.appendChild(row);
    });
  }
})();
