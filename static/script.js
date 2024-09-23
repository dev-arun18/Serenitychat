document.getElementById('chat-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const message = document.getElementById('message').value;
    const chatBox = document.getElementById('chat-box');


    chatBox.innerHTML += `<div class="user-message">You: ${message}</div>`;
    document.getElementById('message').value = '';

    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {

        chatBox.innerHTML += `<div class="bot-message">Bot: ${data.response}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    });
});
