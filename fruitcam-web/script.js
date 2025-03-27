document.addEventListener("DOMContentLoaded", () => {
  const imageInput = document.getElementById("imageInput");
  const uploadBtn = document.getElementById("uploadBtn");
  const uploadedImage = document.getElementById("uploadedImage");
  const predictionResult = document.getElementById("predictionResult");

  // API endpoint - update this with your actual API endpoint
  const API_URL = "http://localhost:8080/api/fruitlens";

  // GIF mapping
  const fruitGifs = {
    'apple': 'assets/apple.gif',
    'banana': 'assets/banana.gif'
  };

  uploadBtn.addEventListener("click", async () => {
    const file = imageInput.files[0];
    if (!file) {
      alert("Please select an image first!");
      return;
    }

    // Display the selected image and prepare the base64 string
    const reader = new FileReader();
    reader.onload = async (e) => {
      // Display the image
      uploadedImage.src = e.target.result;
      uploadedImage.style.display = "block";

      // Get base64 string (remove the data:image/xxx;base64, prefix)
      const base64String = e.target.result.split(",")[1];

      try {
        // Show loading state
        predictionResult.textContent = "Processing...";
        uploadBtn.disabled = true;

        // Send the request to the API
        const response = await fetch(API_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            image: base64String,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        const prediction = result.prediction.toLowerCase();

        // Create HTML for results including the GIF if available
        let resultHTML = `
                    <h3>Prediction Results:</h3>
                    <p>Predicted Fruit: ${result.prediction}</p>
                `;

        if (fruitGifs[prediction]) {
          resultHTML += `
                        <div class="prediction-gif">
                            <img src="${fruitGifs[prediction]}" alt="${prediction} animation" style="max-width: 200px;">
                        </div>
                    `;
        }

        predictionResult.innerHTML = resultHTML;
      } catch (error) {
        console.error("Error:", error);
        predictionResult.textContent =
          "Error: Failed to process the image. Please try again.";
      } finally {
        uploadBtn.disabled = false;
      }
    };

    reader.readAsDataURL(file);
  });
});
