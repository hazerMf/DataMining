document.getElementById("form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const systolicOutput = document.getElementById("systolic_output");
    const diastolicOutput = document.getElementById("diastolic_output");
    
    systolicOutput.style.display = "block";
    diastolicOutput.style.display = "block";
    systolicOutput.querySelector('pre').textContent = "Analyzing systolic blood pressure...";
    diastolicOutput.querySelector('pre').textContent = "Analyzing diastolic blood pressure...";

    try {
        const baseData = {
            Sex: +document.getElementById('Sex').value,
            Age: +document.getElementById('Age').value,
            Height: +document.getElementById('Height').value,
            Weight: +document.getElementById('Weight').value,
            Heart_Rate: +document.getElementById('Heart_Rate').value,
            BMI: +document.getElementById('BMI').value,
            is_raw: false
        };

        // Predict Systolic BP (needs Diastolic BP)
        const systolicData = {
            ...baseData,
            Diastolic_BP: +document.getElementById('Diastolic_BP').value
        };

        // Predict Diastolic BP (needs Systolic BP)
        const diastolicData = {
            ...baseData,
            Systolic_BP: +document.getElementById('Systolic_BP').value
        };

        // Make both predictions in parallel
        const [systolicRes, diastolicRes] = await Promise.all([
            fetch("http://localhost:8000/api/v1/knn/predict/systolic", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(systolicData),
            }),
            fetch("http://localhost:8000/api/v1/knn/predict/diastolic", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(diastolicData),
            })
        ]);

        if (!systolicRes.ok || !diastolicRes.ok) {
            throw new Error(`HTTP error! Systolic status: ${systolicRes.status}, Diastolic status: ${diastolicRes.status}`);
        }

        const [systolicResult, diastolicResult] = await Promise.all([
            systolicRes.json(),
            diastolicRes.json()
        ]);

        // Format and display results
        const formatResult = (result) => {
            return `Predicted Value: ${result.predicted_value_mmHg.toFixed(1)} mmHg
Confidence Interval: ${result.confidence_interval_lower.toFixed(1)} - ${result.confidence_interval_upper.toFixed(1)} mmHg
Standard Deviation: ±${result.prediction_std_mmHg.toFixed(1)} mmHg`;
        };

        systolicOutput.querySelector('pre').textContent = formatResult(systolicResult);
        diastolicOutput.querySelector('pre').textContent = formatResult(diastolicResult);
    } catch (error) {
        outputElement.textContent = `Error: ${error.message}`;
    }
});