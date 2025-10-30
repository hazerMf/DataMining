document.getElementById("form").addEventListener("submit", async (e) => {
      e.preventDefault();

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
        Hypertension: +document.getElementById('Hypertension').value,
        Diabetes_Diabetes: diabetes === "Diabetes",
        Diabetes_None: diabetes === "None",
        Diabetes_Type2: diabetes === "Type2",
        Cerebral_infarction_None: infarction === "None",
        Cerebral_infarction_infarction: infarction === "Infarction",
        Cerebrovascular_None: cvd === "None",
        Cerebrovascular_disease: cvd === "Disease",
        Cerebrovascular_insuff: cvd === "Insuff",
        is_raw: false
      };

      const res = await fetch("http://localhost:8000/api/v1/random-forest/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await res.json();
      document.getElementById("output").textContent = JSON.stringify(result, null, 2);
    });