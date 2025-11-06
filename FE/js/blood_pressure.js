document.getElementById("form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const systolicOutput = document.getElementById("systolic_output");
    const diastolicOutput = document.getElementById("diastolic_output");
    
    systolicOutput.style.display = "block";
    diastolicOutput.style.display = "block";
    systolicOutput.querySelector('pre').textContent = "Analyzing systolic blood pressure...";
    diastolicOutput.querySelector('pre').textContent = "Analyzing diastolic blood pressure...";

    try {
        // Use RAW inputs by default (pre-standardized); backend scalers will be applied
        // compute BMI from height (cm) and weight (kg)
        const heightVal = parseFloat(document.getElementById('Height').value);
        const weightVal = parseFloat(document.getElementById('Weight').value);
        if (!heightVal || heightVal <= 0 || !weightVal || weightVal <= 0) {
            throw new Error('Please enter valid Height (cm) and Weight (kg) to compute BMI');
        }
        const bmiComputed = weightVal / Math.pow((heightVal / 100), 2);
        // show computed BMI in the readonly field for user feedback (if present)
        const bmiField = document.getElementById('BMI');
        if (bmiField) bmiField.value = bmiComputed.toFixed(2);

        const baseData = {
            Sex: +document.getElementById('Sex').value,
            Age: +document.getElementById('Age').value,
            Height: +document.getElementById('Height').value,
            Weight: +document.getElementById('Weight').value,
            Heart_Rate: +document.getElementById('Heart_Rate').value,
            BMI: Number(bmiComputed.toFixed(2)),
            is_raw: true
        };

        // The KNN models require the other BP as an input. To avoid asking the user to provide
        // Systolic/Diastolic we perform an iterative estimate: initialize with population means
        // then alternate predictions until values converge (few iterations).
        let systolic = 120.0; // initial guess (mmHg)
        let diastolic = 80.0; // initial guess (mmHg)

        let systolicResult = null;
        let diastolicResult = null;

        const iterations = 3;
        for (let i = 0; i < iterations; i++) {
            // Predict systolic using current diastolic
            const systolicData = Object.assign({}, baseData, { Diastolic_BP: diastolic });
            const sRes = await fetch("http://localhost:8000/api/v1/knn/predict/systolic", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(systolicData),
            });

            if (!sRes.ok) throw new Error(`Systolic endpoint error: ${sRes.status}`);
            systolicResult = await sRes.json();
            systolic = Number(systolicResult.predicted_value_mmHg);

            // Predict diastolic using updated systolic
            const diastolicData = Object.assign({}, baseData, { Systolic_BP: systolic });
            const dRes = await fetch("http://localhost:8000/api/v1/knn/predict/diastolic", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(diastolicData),
            });

            if (!dRes.ok) throw new Error(`Diastolic endpoint error: ${dRes.status}`);
            diastolicResult = await dRes.json();
            diastolic = Number(diastolicResult.predicted_value_mmHg);
        }

        // Format and display results
        const formatResult = (result) => {
            return `Predicted Value: ${result.predicted_value_mmHg.toFixed(1)} mmHg
Confidence Interval: ${result.confidence_interval_lower.toFixed(1)} - ${result.confidence_interval_upper.toFixed(1)} mmHg
Standard Deviation: ±${result.prediction_std_mmHg.toFixed(1)} mmHg`;
        };

        systolicOutput.querySelector('pre').textContent = formatResult(systolicResult);
        diastolicOutput.querySelector('pre').textContent = formatResult(diastolicResult);
    } catch (error) {
        systolicOutput.querySelector('pre').textContent = `Error: ${error.message}`;
        diastolicOutput.querySelector('pre').textContent = `Error: ${error.message}`;
    }
});