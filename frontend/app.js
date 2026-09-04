const startBtn = document.getElementById('start-btn');
const webcamElement = document.getElementById('webcam');
const recIndicator = document.getElementById('rec-indicator');
const coachChat = document.getElementById('coach-chat');
const sessionScoreEl = document.getElementById('session-score');
const repsCountEl = document.getElementById('reps-count');
const lastScoreEl = document.getElementById('last-score');
const lastMistakeEl = document.getElementById('last-mistake');

let isPracticing = false;
let mediaRecorder;
let recordedChunks = [];
let repCount = 0;
let currentXp = 75;

// ====== SKILL CONFIG ======
let currentSkill = "squat";

const SKILL_CONFIG = {
    squat: {
        icon: "🏋️",
        name: "Bodyweight Squat",
        refVideo: "https://www.youtube.com/embed/eFEVKmp3M4g",
        label: "🏋️ Squat",
        drills: {
            "Shallow":  "<strong style='color:#ff0055'>Issue:</strong> Shallow Depth<br><strong style='color:#00f0ff'>Activity:</strong> Do 3 sets of 10 <b>Goblet Squats</b> with a light kettlebell. Focus on dropping your hips below your knees.",
            "Rounded":  "<strong style='color:#ff0055'>Issue:</strong> Rounded Back<br><strong style='color:#00f0ff'>Activity:</strong> Do 3 sets of 12 <b>Superman Holds</b> to strengthen your lower back. Brace your core before descending.",
            "Unstable": "<strong style='color:#ff0055'>Issue:</strong> Unstable Knees<br><strong style='color:#00f0ff'>Activity:</strong> Wrap a <b>Resistance Band</b> above your knees during squats. Push knees outward against the band.",
            "default":  "<strong style='color:#00f0ff'>Perfect Form!</strong><br>No corrective activities needed. Try adding weights or progress to barbell squats!"
        }
    },
    boxing: {
        icon: "🥊",
        name: "Jab Punch",
        refVideo: "https://www.youtube.com/embed/ghUVhMjay8c",
        label: "🥊 Boxing",
        drills: {
            "Elbow Flared":      "<strong style='color:#ff0055'>Issue:</strong> Elbow Flared<br><strong style='color:#00f0ff'>Activity:</strong> Practice <b>Shadow Boxing</b> with your elbows tucked. Hold a towel between your elbow and ribs to build the habit.",
            "No Hip Rotation":   "<strong style='color:#ff0055'>Issue:</strong> No Hip Rotation<br><strong style='color:#00f0ff'>Activity:</strong> Do 3 sets of 20 <b>Medicine Ball Rotational Throws</b>. Focus on driving the punch from your hips, not your arm.",
            "Guard Dropped":     "<strong style='color:#ff0055'>Issue:</strong> Guard Dropped<br><strong style='color:#00f0ff'>Activity:</strong> Tape a piece of paper to your chin. Practice keeping your non-punching hand touching that paper during every jab.",
            "default":           "<strong style='color:#00f0ff'>Clean Jab!</strong><br>Great technique! Work on speed drills: 3 rounds of 30-second rapid jabs on the heavy bag."
        }
    },
    basketball: {
        icon: "🏀",
        name: "Free Throw Shot",
        refVideo: "https://www.youtube.com/embed/t7Gx4QYq9Bw",
        label: "🏀 Basketball",
        drills: {
            "Low Elbow":          "<strong style='color:#ff0055'>Issue:</strong> Low Elbow<br><strong style='color:#00f0ff'>Activity:</strong> Stand against a wall, raise your shooting elbow to 90°, and practice the <b>One-Hand Form Shooting Drill</b> from 3 feet.",
            "No Leg Drive":       "<strong style='color:#ff0055'>Issue:</strong> No Leg Drive<br><strong style='color:#00f0ff'>Activity:</strong> Do 3 sets of 15 <b>Box Jumps</b> to build explosive leg power. Bend your knees deeply before every shot.",
            "Poor Release Angle": "<strong style='color:#ff0055'>Issue:</strong> Poor Release Angle<br><strong style='color:#00f0ff'>Activity:</strong> Lie flat on your back and shoot straight up. The ball should land back in your hands. This forces a high arc release.",
            "default":            "<strong style='color:#00f0ff'>Splash!</strong><br>Your shot mechanics look great. Focus on consistency: shoot 50 free throws from the line."
        }
    }
};

// ====== SKILL SELECTION ======
document.querySelectorAll('.skill-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        if (isPracticing) {
            addCoachMessage("⚠️ Stop your current practice before switching skills!");
            return;
        }

        document.querySelectorAll('.skill-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentSkill = tab.dataset.skill;

        const config = SKILL_CONFIG[currentSkill];
        document.getElementById('skill-icon-display').textContent = config.icon;
        document.getElementById('skill-name-display').textContent = config.name;
        document.getElementById('ref-video').src = config.refVideo;
        document.getElementById('active-skill-label').textContent = config.label;
        document.getElementById('drill-text').innerHTML = "Complete a rep to get personalized drill recommendations!";

        // Reset counters on skill switch
        repCount = 0;
        currentXp = 75;
        repsCountEl.textContent = 0;
        sessionScoreEl.textContent = "--";
        lastScoreEl.textContent = "--";
        lastMistakeEl.textContent = "None";
        document.getElementById('xp-fill').style.width = "75%";
        document.getElementById('xp-text').textContent = "Level 4 (75%)";

        addCoachMessage(`Switched to ${config.name}! Hit Start Practice when you're ready.`);
    });
});

// ====== CHAT ======
const queryInput = document.getElementById('user-query');
const sendBtn = document.getElementById('send-query');

sendBtn.addEventListener('click', () => {
    const text = queryInput.value.trim();
    if (!text) return;
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user-msg';
    msgDiv.innerHTML = `<p>${text}</p>`;
    coachChat.appendChild(msgDiv);
    queryInput.value = '';
    coachChat.scrollTop = coachChat.scrollHeight;

    setTimeout(() => {
        addCoachMessage("Great question! In a production version, I'd use the Gemini API to give you a personalized answer based on your skill data and history.");
    }, 1000);
});

queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendBtn.click();
});

// ====== CAMERA ======
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" },
            audio: false
        });
        webcamElement.srcObject = stream;
        webcamElement.onloadedmetadata = () => {
            const canvas = document.getElementById('overlay');
            if (canvas) {
                canvas.width = webcamElement.videoWidth;
                canvas.height = webcamElement.videoHeight;
            }
        };
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) recordedChunks.push(event.data);
        };
        mediaRecorder.onstop = sendVideoToBackend;
    } catch (error) {
        console.error("Camera access denied:", error);
        addCoachMessage("I couldn't access your camera. Please allow permissions to start.");
    }
}

// ====== PRACTICE TOGGLE ======
startBtn.addEventListener('click', () => {
    isPracticing = !isPracticing;
    if (isPracticing) {
        startBtn.textContent = "Stop Practice";
        startBtn.style.background = "var(--secondary)";
        startBtn.style.boxShadow = "0 0 15px var(--secondary)";
        recIndicator.style.display = "flex";
        const config = SKILL_CONFIG[currentSkill];
        addCoachMessage(`Let's go! I'm watching your ${config.name} form. Show me what you've got!`);
        startRecordingCycle();
    } else {
        startBtn.textContent = "Start Practice";
        startBtn.style.background = "var(--primary)";
        startBtn.style.boxShadow = "0 0 15px var(--primary-glow)";
        recIndicator.style.display = "none";
        addCoachMessage("Practice paused. Great work!");
    }
});

function startRecordingCycle() {
    if (!isPracticing) return;
    recordedChunks = [];
    mediaRecorder.start();
    setTimeout(() => {
        if (mediaRecorder.state === "recording") mediaRecorder.stop();
    }, 4000);
}

// ====== SEND TO BACKEND ======
async function sendVideoToBackend() {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const formData = new FormData();
    formData.append('file', blob, 'rep.webm');
    formData.append('skill', currentSkill);

    addCoachMessage("Analyzing your last rep...");

    try {
        const response = await fetch('http://127.0.0.1:8000/evaluate', {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error("Backend analysis failed");
        const result = await response.json();
        handleAnalysisResult(result);
    } catch (error) {
        console.error(error);
        addCoachMessage("Hmm, I couldn't analyze that rep. Make sure your body is in frame!");
    }

    if (isPracticing) setTimeout(startRecordingCycle, 1000);
}

// ====== HANDLE RESULTS ======
function handleAnalysisResult(result) {
    if (result.error) { addCoachMessage(result.error); return; }

    repCount++;
    repsCountEl.textContent = repCount;
    sessionScoreEl.textContent = result.overall_score;
    lastScoreEl.textContent = result.overall_score;
    lastMistakeEl.textContent = result.detected_mistake;

    // XP & Leveling
    if (result.overall_score >= 85) currentXp += 15;
    else if (result.overall_score >= 70) currentXp += 5;

    if (currentXp >= 100) {
        currentXp = currentXp - 100;
        document.getElementById('xp-text').textContent = `Level 5 (${currentXp}%)`;
        document.getElementById('xp-fill').style.width = `${currentXp}%`;
        addCoachMessage("🎉 LEVEL UP! You've reached Level 5! Keep pushing!");
    } else {
        document.getElementById('xp-text').textContent = `Level 4 (${currentXp}%)`;
        document.getElementById('xp-fill').style.width = `${currentXp}%`;
    }

    // Skill-specific Drills
    const config = SKILL_CONFIG[currentSkill];
    const drillTextEl = document.getElementById('drill-text');
    let drillFound = false;
    for (const [keyword, html] of Object.entries(config.drills)) {
        if (keyword !== "default" && result.detected_mistake.includes(keyword)) {
            drillTextEl.innerHTML = html;
            drillFound = true;
            break;
        }
    }
    if (!drillFound) drillTextEl.innerHTML = config.drills["default"];

    // Coach Feedback
    if (result.overall_score > 90) {
        addCoachMessage(`Score: ${result.overall_score}! 🔥 Perfect ${config.name} form! Keep it up!`);
    } else if (result.overall_score > 70) {
        addCoachMessage(`Score: ${result.overall_score}. Good job, but I noticed: ${result.detected_mistake}. Focus on that next rep.`);
    } else {
        addCoachMessage(`Score: ${result.overall_score}. Careful! ${result.detected_mistake}. Slow down and focus on form.`);
    }
}

function addCoachMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message ai-msg';
    msgDiv.innerHTML = `<p>${text}</p>`;
    coachChat.appendChild(msgDiv);
    coachChat.scrollTop = coachChat.scrollHeight;
}

// Init
setupCamera();
