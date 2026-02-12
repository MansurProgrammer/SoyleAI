Project Title: SoyleAI

***

Project Description: Modern language learning tool SoyleAI helps improve the pronunciation of distinctive Kazakh letters (Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, Һ, І) using AI. Unlike most language learning tools, SoyleAI examines the physical aspects of talk, or articulation.

***

Key Technologies:

Computer Vision (MediaPipe): It uses computer vision to track the facial landmarks in real time and observe the geometry of the lips and the openness of the

Neural Networks (Wav2Vec 2.0): This works by analyzing the signal to ensure phonetic accuracy.

PyQt6: Offers an intuitive interface.

***

How it Works:

Visual Tracking: The system keeps track of the user's mouth movements. For instance, for the character "Ә," it checks whether the mouth is opened sufficiently.

Sound Validation: The AI checks the user’s voice against a phonetic model.

Smart Feedback: A percentage is provided, as well as specific anatomical feedback (e.g., "Open your mouth wider"). Goal: "Speech Therapist" is used to describe our digital, 24/7 facilitation of students and beginners learning Kazakh through modern technology.

***

How to run?
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python app.py`

*Note: Ensure the 'assets' folder and model files (.pkl) are in the same directory as app.py.*

