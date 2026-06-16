const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const langSelect = document.getElementById('language-selector');

// Dynamic greetings based on language
const greetings = {
    'en': 'Hello! I am your AI assistant. How can I help you with your studies today?',
    'si': 'ආයුබෝවන්! මම ඔබේ AI සහායකයා වෙමි. අද ඔබේ අධ්‍යයන කටයුතු සඳහා මට උපකාර කළ හැක්කේ කෙසක්ද?',
    'ta': 'வணக்கம்! நான் உங்கள் AI உதவியாளர். இன்று உங்கள் படிப்பிற்கு நான் எப்படி உதவ முடியும்?'
};

langSelect.addEventListener('change', () => {
    chatBox.innerHTML = '';
    appendMessage(greetings[langSelect.value], 'bot');
});

function appendMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', `${sender}-message`);
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.classList.add('msg-bubble');
    bubbleDiv.innerHTML = text;
    
    msgDiv.appendChild(bubbleDiv);
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function showTyping() {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', 'bot-message');
    msgDiv.id = 'typing-indicator-msg';
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.classList.add('msg-bubble');
    bubbleDiv.innerHTML = `
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>
    `;
    
    msgDiv.appendChild(bubbleDiv);
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function hideTyping() {
    const el = document.getElementById('typing-indicator-msg');
    if (el) el.remove();
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;
    
    const lang = langSelect.value;
    const model = document.getElementById('model-selector').value;
    appendMessage(text, 'user');
    userInput.value = '';
    showTyping();
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, language: lang, model: model })
        });
        
        const data = await response.json();
        hideTyping();
        setTimeout(() => { appendMessage(data.response, 'bot'); }, 300);
    } catch (error) {
        hideTyping();
        appendMessage('Connection error. Please ensure the API server is running.', 'bot');
    }
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// --- Recommendation Logic ---
const recModal = document.getElementById('recommend-modal');
const openRecBtn = document.getElementById('recommend-btn');
const closeRecBtn = document.getElementById('close-modal');
const getRecBtn = document.getElementById('get-rec-btn');
const recInput = document.getElementById('rec-input');
const recResults = document.getElementById('rec-results');

openRecBtn.addEventListener('click', () => { recModal.style.display = 'flex'; });
closeRecBtn.addEventListener('click', () => { recModal.style.display = 'none'; });

async function fetchRecommendations() {
    const interests = recInput.value.trim();
    if (!interests) return;
    
    recResults.innerHTML = '<p style="text-align:center; color: var(--text-muted);">Finding the best courses...</p>';
    
    try {
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ interests: interests })
        });
        
        const data = await response.json();
        recResults.innerHTML = '';
        
        if (!data.recommendations || data.recommendations.length === 0) {
            recResults.innerHTML = '<p style="text-align:center;">No courses found matching your interests.</p>';
            return;
        }
        
        data.recommendations.forEach(course => {
            const card = document.createElement('div');
            card.className = 'course-card';
            
            const tagsHtml = course.tags.map(tag => `<span class="tag">${tag}</span>`).join('');
            
            card.innerHTML = `
                <h4>${course.title} <span style="font-size: 0.7rem; color: #00e676; float:right;">${(course.similarity_score * 100).toFixed(0)}% Match</span></h4>
                <p style="margin-bottom: 8px;">${course.description}</p>
                <div class="course-tags">${tagsHtml}</div>
            `;
            recResults.appendChild(card);
        });
        
    } catch (error) {
        recResults.innerHTML = '<p style="text-align:center; color: red;">Error fetching recommendations.</p>';
    }
}

getRecBtn.addEventListener('click', fetchRecommendations);
recInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') fetchRecommendations();
});

// --- Slide Generator Logic ---
const slidesModal = document.getElementById('slides-modal');
const openSlidesBtn = document.getElementById('slides-btn');
const closeSlidesBtn = document.getElementById('close-slides');
const getSlidesBtn = document.getElementById('get-slides-btn');
const slideTopicInput = document.getElementById('slide-topic-input');
const presentationView = document.getElementById('presentation-view');

const slideTitle = document.getElementById('slide-title');
const slideContent = document.getElementById('slide-content');
const prevSlideBtn = document.getElementById('prev-slide');
const nextSlideBtn = document.getElementById('next-slide');
const slideCounter = document.getElementById('slide-counter');

let currentSlides = [];
let currentSlideIndex = 0;

openSlidesBtn.addEventListener('click', () => { slidesModal.style.display = 'flex'; });
closeSlidesBtn.addEventListener('click', () => { 
    slidesModal.style.display = 'none'; 
    presentationView.style.display = 'none';
    slideTopicInput.value = '';
});

function updateSlideView() {
    if (currentSlides.length === 0) return;
    
    const slide = currentSlides[currentSlideIndex];
    slideTitle.textContent = slide.title;
    slideContent.innerHTML = slide.content;
    
    slideCounter.textContent = `${currentSlideIndex + 1} / ${currentSlides.length}`;
    
    prevSlideBtn.style.opacity = currentSlideIndex === 0 ? '0.5' : '1';
    prevSlideBtn.style.pointerEvents = currentSlideIndex === 0 ? 'none' : 'auto';
    
    nextSlideBtn.style.opacity = currentSlideIndex === currentSlides.length - 1 ? '0.5' : '1';
    nextSlideBtn.style.pointerEvents = currentSlideIndex === currentSlides.length - 1 ? 'none' : 'auto';
}

async function fetchSlides() {
    const topic = slideTopicInput.value.trim();
    if (!topic) return;
    
    getSlidesBtn.innerHTML = '...';
    getSlidesBtn.disabled = true;
    
    try {
        const response = await fetch('/generate-slides', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic: topic, language: langSelect.value })
        });
        
        const data = await response.json();
        currentSlides = data.slides || [];
        currentSlideIndex = 0;
        
        if (currentSlides.length > 0) {
            presentationView.style.display = 'flex';
            updateSlideView();
        } else {
            alert('No slides could be generated for this topic.');
        }
        
    } catch (error) {
        alert('Error fetching slides.');
    } finally {
        getSlidesBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h20v14H2z"></path><path d="M8 21h8"></path><path d="M12 17v4"></path></svg>';
        getSlidesBtn.disabled = false;
    }
}

getSlidesBtn.addEventListener('click', fetchSlides);
slideTopicInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') fetchSlides();
});

prevSlideBtn.addEventListener('click', () => {
    if (currentSlideIndex > 0) {
        currentSlideIndex--;
        updateSlideView();
    }
});

nextSlideBtn.addEventListener('click', () => {
    if (currentSlideIndex < currentSlides.length - 1) {
        currentSlideIndex++;
        updateSlideView();
    }
});
