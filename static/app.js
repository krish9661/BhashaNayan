function updateWords() {
    fetch("/get_words")
        .then(response => response.json())
        .then(data => {
            document.getElementById("recognized-words").innerText = data.words.join(" ");
        })
        .catch(err => console.error("Error fetching words:", err));
}

function clearWords() {
    // Clear the text in the box
    document.getElementById("recognized-words").innerText = "Waiting for gestures...";

    // Optional: Reset the recognized words list on the backend
    fetch("/clear_words", { method: "POST" })
        .then(response => {
            if (!response.ok) {
                throw new Error("Failed to clear words on the server");
            }
            console.log("Recognized words cleared");
        })
        .catch(err => console.error("Error clearing words:", err));
}

// Attach the clear function to the button
document.getElementById("clear-button").addEventListener("click", clearWords);

// Update words every 500ms
setInterval(updateWords, 500);