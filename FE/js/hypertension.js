document.getElementById("form").addEventListener("submit", async (e) => {
      e.preventDefault();

  const outputElement = document.getElementById("output");
  outputElement.style.display = "block";
  outputElement.innerHTML = "<strong>Analyzing...</strong>";

      try {
        const diabetes = document.querySelector('input[name="Diabetes"]:checked').value;
        const infarction = document.querySelector('input[name="Infarction"]:checked').value;
        const cvd = document.querySelector('input[name="CVD"]:checked').value;

        // compute BMI from height (cm) and weight (kg)
        const heightVal = parseFloat(document.getElementById('Height').value);
        const weightVal = parseFloat(document.getElementById('Weight').value);
        if (!heightVal || heightVal <= 0 || !weightVal || weightVal <= 0) {
          throw new Error('Please enter valid Height (cm) and Weight (kg) to compute BMI');
        }
        if ( !document.getElementById('Sex').value || !document.getElementById('Age').value){
          throw new Error('Please enter Gender and Age');
        }
        if ( !document.getElementById('Systolic_BP').value || !document.getElementById('Diastolic_BP').value || !document.getElementById('Heart_Rate').value ){
          throw new Error('Please enter valid Systolic BP, Diastolic BP and heart rate');
        }
        const bmiComputed = weightVal / Math.pow((heightVal / 100), 2);
        // show computed BMI in the readonly field for user feedback
        const bmiField = document.getElementById('BMI');
        if (bmiField) bmiField.value = bmiComputed.toFixed(2);

        const data = {
          Sex: +document.getElementById('Sex').value,
          Age: +document.getElementById('Age').value,
          Height: +document.getElementById('Height').value,
          Weight: +document.getElementById('Weight').value,
          Systolic_BP: +document.getElementById('Systolic_BP').value,
          Diastolic_BP: +document.getElementById('Diastolic_BP').value,
          Heart_Rate: +document.getElementById('Heart_Rate').value,
          BMI: Number(bmiComputed.toFixed(2)),
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