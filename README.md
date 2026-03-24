Project Title: SoyleAI

***

Project Description: Modern language learning tool SoyleAI helps improve the pronunciation of distinctive Kazakh letters (Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, Һ, І) using AI. Unlike most language learning tools, SoyleAI examines the physical aspects of talk, or articulation.

***

Key Technologies:

* **Python 3.11**
* **Mediapipe** (Face Mesh for lip tracking)
* **PyTorch & Transformers** (Wav2Vec2 for audio analysis)
* **Scikit-learn** (Classification models)
* **PyQt6** (Modern Desktop UI)

***

How it Works:

Visual Tracking: The system keeps track of the user's mouth movements. For instance, for the character "Ә," it checks whether the mouth is opened sufficiently.

Sound Validation: The AI checks the user’s voice against a phonetic model.

Smart Feedback: A percentage is provided, as well as specific anatomical feedback (e.g., "Open your mouth wider"). Goal: "Speech Therapist" is used to describe our digital, 24/7 facilitation of students and beginners learning Kazakh through modern technology.

Images adapted from Pronuncian.com for phonetic guidance.

***

How to run?
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python main.py`

*Note: Ensure the 'assets' folder and model files (.pkl) are in the same directory as app.py.*
*On the first run, the app will download the Wav2Vec2 base model (approx. 300MB). Please ensure you have a stable internet connection.*






