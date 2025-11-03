document.getElementById("form").addEventListener("submit", async (e) => {
      e.preventDefault();

  const outputElement = document.getElementById("output");
  outputElement.style.display = "block";
  outputElement.innerHTML = "<strong>Analyzing...</strong>";

      try {
        const diabetes = document.querySelector('input[name="Diabetes"]:checked').value;
        const infarction = document.querySelector('input[name="Infarction"]:checked').value;
        const cvd = document.querySelector('input[name="CVD"]:checked').value;

        const data = {
          Sex: +document.getElementById('Sex').value,
          Age: +document.getElementById('Age').value,
          Height: +document.getElementById('Height').value,
          Weight: +document.getElementById('Weight').value,
          Systolic_BP: +document.getElementById('Systolic_BP').value,
          Diastolic_BP: +document.getElementById('Diastolic_BP').value,
          Heart_Rate: +document.getElementById('Heart_Rate').value,
          BMI: +document.getElementById('BMI').value,
          Diabetes_Diabetes: diabetes === "Diabetes" ? 1 : 0,
          Diabetes_None: diabetes === "None" ? 1 : 0,
          Diabetes_Type2: diabetes === "Type2" ? 1 : 0,
          Cerebral_infarction_None: infarction === "None" ? 1 : 0,
          Cerebral_infarction_infarction: infarction === "Infarction" ? 1 : 0,
          Cerebrovascular_None: cvd === "None" ? 1 : 0,
          Cerebrovascular_disease: cvd === "Disease" ? 1 : 0,
          Cerebrovascular_insuff: cvd === "Insuff" ? 1 : 0,
          // Interpret values as RAW (pre-standardized) by default
          is_raw: true
        };

        const res = await fetch("http://localhost:8000/api/v1/random-forest/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });

        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }

        const result = await res.json();

        // Render a friendly result card instead of raw JSON
        const prob = (result.probability || 0) * 100;
        outputElement.innerHTML = `
          <div class="w3-panel w3-pale-yellow w3-leftbar w3-border-yellow w3-round-large">
            <h3 style="margin-top:0">Prediction: ${result.label}</h3>
            <p style="margin:0.25rem 0">Confidence: <strong>${prob.toFixed(1)}%</strong></p>
            <p class="w3-small" style="color:#475569; margin-top:8px">This prediction used the Random Forest model and treated your inputs as raw (is_raw=true). Interpret results as guidance, not medical advice.</p>
          </div>
        `;
      } catch (error) {
        outputElement.innerHTML = `<div class="w3-panel w3-pale-red w3-round-large">Error: ${error.message}</div>`;
      }
    });