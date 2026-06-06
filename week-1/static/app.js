document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");
    const btnReset = document.getElementById("btnReset");
    
    const resultsPlaceholder = document.getElementById("resultsPlaceholder");
    const resultsContent = document.getElementById("resultsContent");
    const resultsLoader = document.getElementById("resultsLoader");
    
    const riskDial = document.getElementById("riskDial");
    const riskProbability = document.getElementById("riskProbability");
    const riskBadge = document.getElementById("riskBadge");
    const riskLevelText = document.getElementById("riskLevelText");
    const modelPrediction = document.getElementById("modelPrediction");
    const recommendationsList = document.getElementById("recommendationsList");
    
    const DIAL_CIRCUMFERENCE = 565.48;
    
    riskDial.style.strokeDasharray = DIAL_CIRCUMFERENCE;
    riskDial.style.strokeDashoffset = DIAL_CIRCUMFERENCE;
    
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        resultsPlaceholder.classList.add("hidden");
        resultsContent.classList.add("hidden");
        resultsLoader.classList.remove("hidden");
        
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = parseFloat(value);
        });
        
        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            setTimeout(() => {
                resultsLoader.classList.add("hidden");
                
                if (response.ok) {
                    renderResults(result);
                } else {
                    showError(result.error || "An error occurred during assessment.");
                }
            }, 600);
            
        } catch (err) {
            console.error("Fetch Error:", err);
            resultsLoader.classList.add("hidden");
            showError("Unable to connect to prediction server.");
        }
    });
    
    btnReset.addEventListener("click", () => {
        form.reset();
        resetResults();
    });
    
    function renderResults(data) {
        resultsContent.classList.remove("hidden");
        
        riskProbability.textContent = `${data.probability}%`;
        riskLevelText.textContent = `${data.risk_level} Risk`;
        modelPrediction.textContent = data.prediction === 1 
            ? "Diabetes Detected (1)" 
            : "No Diabetes (0)";
        
        const offset = DIAL_CIRCUMFERENCE - (DIAL_CIRCUMFERENCE * data.probability) / 100;
        riskDial.style.stroke = data.color;
        riskDial.style.strokeDashoffset = offset;
        
        riskDial.style.filter = `drop-shadow(0 0 8px ${data.color})`;
        
        riskBadge.style.backgroundColor = `${data.color}15`;
        riskBadge.style.color = data.color;
        riskBadge.style.border = `1px solid ${data.color}40`;
        
        recommendationsList.innerHTML = "";
        data.recommendations.forEach((rec) => {
            const li = document.createElement("li");
            li.textContent = rec;
            li.style.setProperty('--icon-color', data.color);
            recommendationsList.appendChild(li);
        });
    }
    
    function resetResults() {
        resultsContent.classList.add("hidden");
        resultsLoader.classList.add("hidden");
        resultsPlaceholder.classList.remove("hidden");
        
        riskDial.style.strokeDashoffset = DIAL_CIRCUMFERENCE;
        riskDial.style.stroke = "#10B981";
        riskDial.style.filter = "none";
    }
    
    function showError(message) {
        resultsPlaceholder.classList.remove("hidden");
        resultsPlaceholder.innerHTML = `
            <div class="placeholder-icon-container" style="border-color: rgba(239, 68, 68, 0.2)">
                <i class="fa-solid fa-triangle-exclamation placeholder-icon" style="color: #EF4444"></i>
            </div>
            <h3 style="color: #EF4444">Assessment Failed</h3>
            <p style="max-width: 320px">${message}</p>
            <button onclick="window.location.reload()" class="btn btn-secondary" style="margin-top: 15px; padding: 8px 16px; font-size: 0.85rem">
                Retry
            </button>
        `;
    }
});
