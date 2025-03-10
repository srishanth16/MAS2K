import React, { useState } from "react";

function App() {
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [predictedDisease, setPredictedDisease] = useState("");

  const symptoms = ["cough", "fatigue", "fever", "headache", "nausea"];

  const handleCheckboxChange = (symptom) => {
    setSelectedSymptoms((prev) =>
      prev.includes(symptom)
        ? prev.filter((s) => s !== symptom)
        : [...prev, symptom]
    );
  };

  const handleSubmit = async () => {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ symptoms: selectedSymptoms }),
    });

    const data = await response.json();
    setPredictedDisease(data.predicted_disease);
  };

  return (
    <div>
      <h1>Disease Prediction</h1>
      {symptoms.map((symptom) => (
        <label key={symptom}>
          <input
            type="checkbox"
            value={symptom}
            onChange={() => handleCheckboxChange(symptom)}
          />
          {symptom}
        </label>
      ))}
      <button onClick={handleSubmit}>Predict Disease</button>
      {predictedDisease && <h2>Predicted Disease: {predictedDisease}</h2>}
    </div>
  );
}

export default App;
