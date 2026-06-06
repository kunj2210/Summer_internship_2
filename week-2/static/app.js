document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    
    // Output DOM elements
    const predictedPriceEl = document.getElementById('predicted-price');
    const rfPriceEl = document.getElementById('rf-price');
    const lrPriceEl = document.getElementById('lr-price');
    const modelDescriptorEl = document.getElementById('model-descriptor');
    const priceDeltaEl = document.getElementById('price-delta');
    const deltaBar = document.getElementById('delta-bar');
    const recsList = document.getElementById('recommendations-list');
    const imputationAlert = document.getElementById('imputation-alert');
    const imputationAlertText = document.getElementById('imputation-alert-text');
    
    // Model cards
    const rfCard = document.getElementById('rf-card');
    const lrCard = document.getElementById('lr-card');
    
    // Click events on model cards to switch between preferred outputs
    let currentPredictionData = null;
    let selectedModel = 'rf'; // default to Random Forest (best R²)

    rfCard.addEventListener('click', () => {
        if (!currentPredictionData) return;
        selectedModel = 'rf';
        rfCard.classList.add('active');
        lrCard.classList.remove('active');
        updateActiveDisplay();
    });

    lrCard.addEventListener('click', () => {
        if (!currentPredictionData) return;
        selectedModel = 'lr';
        lrCard.classList.add('active');
        rfCard.classList.remove('active');
        updateActiveDisplay();
    });

    function updateActiveDisplay() {
        if (!currentPredictionData) return;
        
        let price = 0;
        let descriptor = "";
        
        if (selectedModel === 'rf') {
            price = currentPredictionData.random_forest_price;
            descriptor = "Based on Random Forest model (R²: 83.40%)";
        } else {
            price = currentPredictionData.linear_regression_price;
            descriptor = "Based on Linear Regression model (R²: 80.77%)";
        }
        
        animatePriceCountUp(price);
        modelDescriptorEl.textContent = descriptor;
    }

    // Number animation utility
    function animatePriceCountUp(targetVal) {
        let current = 0;
        const duration = 800; // ms
        const stepTime = 16; // ms (~60fps)
        const totalSteps = duration / stepTime;
        const increment = targetVal / totalSteps;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= targetVal) {
                current = targetVal;
                clearInterval(timer);
            }
            predictedPriceEl.textContent = current.toFixed(2);
        }, stepTime);
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Add loading state
        document.querySelector('.price-showcase').classList.add('loading');
        submitBtn.disabled = true;
        submitBtn.querySelector('span').textContent = 'Processing Valuation...';
        
        // Extract form data
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            // Send empty strings as null so that the backend can impute them
            if (value === "") {
                data[key] = null;
            } else {
                data[key] = value;
            }
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            
            if (!response.ok) {
                alert(result.error || 'Server error occurred during prediction.');
                return;
            }
            
            currentPredictionData = result;
            
            // 1. Update Sub-Model display values
            rfPriceEl.textContent = `₹ ${result.random_forest_price.toFixed(2)} L`;
            lrPriceEl.textContent = `₹ ${result.linear_regression_price.toFixed(2)} L`;
            
            // 2. Set active display values
            updateActiveDisplay();
            
            // 3. Show Delta (Difference) between models
            priceDeltaEl.textContent = `₹ ${result.difference.toFixed(2)} Lakhs`;
            // Calculate a ratio for the delta progress bar relative to average price
            const avgPrice = (result.random_forest_price + result.linear_regression_price) / 2;
            const deltaRatio = Math.min((result.difference / avgPrice) * 100, 100);
            deltaBar.style.width = `${deltaRatio}%`;
            
            // 4. Render Imputation alert if backend filled blank values
            if (result.mileage_imputed || result.engine_imputed) {
                let msg = "Notice: Missing fields filled using dataset medians. ";
                if (result.mileage_imputed) {
                    msg += `Mileage auto-set to ${result.imputed_mileage_val.toLocaleString()} km. `;
                }
                if (result.engine_imputed) {
                    msg += `Engine Size auto-set to ${result.imputed_engine_val} cc. `;
                }
                imputationAlertText.textContent = msg;
                imputationAlert.classList.remove('hidden');
            } else {
                imputationAlert.classList.add('hidden');
            }
            
            // 5. Update Recommendations List
            recsList.innerHTML = '';
            result.recommendations.forEach(rec => {
                const li = document.createElement('li');
                li.textContent = rec;
                recsList.appendChild(li);
            });
            
        } catch (error) {
            console.error('Error:', error);
            alert('Could not connect to server. Please ensure the backend Flask server is running.');
        } finally {
            document.querySelector('.price-showcase').classList.remove('loading');
            submitBtn.disabled = false;
            submitBtn.querySelector('span').textContent = 'Evaluate Car Value';
        }
    });
});
