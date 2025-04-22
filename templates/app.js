function updateWords() {
    fetch("/get_words")
        .then(response => response.json())
        .then(data => {
            document.getElementById("recognized-words").innerText = data.words.join(" ");
        })
        .catch(err => console.error("Error fetching words:", err));
}
// Update words every 500ms
setInterval(updateWords, 500);