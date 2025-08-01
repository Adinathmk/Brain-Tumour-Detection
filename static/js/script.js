// Get references to the DOM elements
const dropArea = document.getElementById("drop-area");
const inputFile = document.getElementById("input-file");
const imageView = document.getElementById("img-view");
const uploadButton = document.getElementById("upload-button");

let selectedFile = null; // This will hold the currently selected file

// When the user selects a file via the file input...
inputFile.addEventListener("change", function () {
    if (inputFile.files.length > 0) {
        selectedFile = inputFile.files[0];
        // Use FileReader to display a preview of the image
        const reader = new FileReader();
        reader.onload = function (e) {
            // Set the preview background image using the Data URL
            imageView.style.backgroundImage = `url(${e.target.result})`;
            // Clear any placeholder text and border
            imageView.textContent = "";
            imageView.style.border = "none";
        };
        reader.readAsDataURL(selectedFile);
    } else {
        alert("No file selected! Please upload an image.");
    }
});

// Allow drag & drop on the designated drop area
dropArea.addEventListener("dragover", function (e) {
    e.preventDefault(); // Prevent default behavior (open as link for some elements)
});

dropArea.addEventListener("drop", function (e) {
    e.preventDefault();
    // Set the dropped files to the file input element
    inputFile.files = e.dataTransfer.files;
    // Manually trigger the change event so the preview is updated
    const event = new Event('change');
    inputFile.dispatchEvent(event);
});

// When the user clicks the "Upload Image" button...
uploadButton.addEventListener("click", function () {
    if (!selectedFile) {
        alert("Please select an image first.");
        return;
    }

    // Create a loading animation inside the button
    uploadButton.innerHTML = `<img src="static/img/loading.gif" style="width: 20px; height: 20px; margin-right: 8px;"> Uploading...`;
    uploadButton.disabled = true; // Disable button to prevent multiple clicks

    // Create FormData to send the file to Flask
    const formData = new FormData();
    formData.append("file", selectedFile);

    // Send the file to Flask using Fetch API
    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.text())  // Expecting a redirected HTML response
    .then(html => {
        document.open();
        document.write(html);
        document.close();
    })
    .catch(error => {
        console.error("Error uploading file:", error);
        alert("Error uploading file. Please try again.");
    })
    .finally(() => {
        // Restore button state after upload (optional)
        uploadButton.innerHTML = "Upload Image";
        uploadButton.disabled = false;
    });
});
