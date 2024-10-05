let conversation = [];

function sendMessage() {
    const userInput = document.getElementById('userInput').value;
    appendMessage(userInput, 'user');
    conversation.push({ sender: 'user', message: userInput });
    document.getElementById('userInput').value = '';

    // Send user input to the backend
    fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        const aiResponse = data.response;
        appendMessage(aiResponse, 'ai');
        conversation.push({ sender: 'ai', message: aiResponse });
    });
}

function appendMessage(message, sender) {
    const chatBox = document.getElementById('chatBox');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}`;
    messageDiv.innerText = message;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function submitConversation() {
    const conversationData = JSON.stringify(conversation);

    fetch('/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation: conversationData })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            window.location.href = `/results?data=${encodeURIComponent(conversationData)}`;
        } else {
            alert('Error saving conversation: ' + data.message);
        }
    });
}

function adjustInputHeight() {
    const textarea = document.getElementById('userInput');
    textarea.style.height = 'auto';
    textarea.style.height = (textarea.scrollHeight) + 'px';
}

function checkEnter(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
    }
}

function startRecognition() {
    const pulseButton = document.getElementById('pulse');
    const micIcon = document.getElementById('mic-icon');
    pulseButton.classList.add('pulsing');

    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
        const speechResult = event.results[0][0].transcript;
        appendMessage(speechResult, 'user');
        conversation.push({ sender: 'user', message: speechResult });

        // Send speech result to the backend
        fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: speechResult })
        })
        .then(response => response.json())
        .then(data => {
            const aiResponse = data.response;
            appendMessage(aiResponse, 'ai');
            conversation.push({ sender: 'ai', message: aiResponse });
        });

        // Stop pulsing after processing the result
        pulseButton.classList.remove('pulsing');
        micIcon.classList.remove('fa-microphone'); // Change icon to microphone-slash
        micIcon.classList.add('fa-microphone-slash');
    };

    recognition.onstart = () => {
        // Ensure the button keeps pulsing while listening
        pulseButton.classList.add('pulsing');
        micIcon.classList.remove('fa-microphone-slash'); // Change icon to microphone-lines
        micIcon.classList.add('fa-microphone');
    };

    recognition.onend = () => {
        // Remove the pulsing effect when recognition ends
        pulseButton.classList.remove('pulsing');
        micIcon.classList.remove('fa-microphone'); // Change icon back to microphone-slash
        micIcon.classList.add('fa-microphone-slash');
    };

    recognition.start();
}