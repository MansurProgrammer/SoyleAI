import sys
import os
import json
import librosa
import pickle
import difflib
import queue
import numpy as np
import sounddevice as sd
import cv2
import mediapipe as mp
from datetime import datetime
from vosk import Model, KaldiRecognizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QComboBox, QHBoxLayout, QFrame, QStackedWidget, QScrollArea, QGridLayout, QGraphicsOpacityEffect, QGraphicsBlurEffect)
from PyQt6.QtGui import (QFont, QImage, QPixmap, QPainter, QColor, QPen, QMovie, QPainterPath, QLinearGradient)
from PyQt6.QtCore import (Qt, QTimer, QRectF, QVariantAnimation, QEasingCurve, QThread, pyqtSignal, QUrl, QPropertyAnimation, QPointF)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

SETTINGS_FILE = "settings.json"
DB_FILE = "history.json"
MODEL_FILE = "model_data.pkl"
VOSK_MODEL_PATH = "model_kz" 

def format_display_name(folder_name): 
    return folder_name.upper()

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            pass
    return {"auto_save": False}

def save_settings(settings_dict):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as file:
        json.dump(settings_dict, file, ensure_ascii=False, indent=4)

CONFUSION_GROUPS = {
    'Е': ['Э'], 'Э': ['Е'], 'Қ': ['К', 'Х'], 'К': ['Қ', 'Х'], 'Х': ['Қ', 'К'], 
    'С': ['Ш'], 'Ш': ['С'], 'Б': ['П', 'В'], 'П': ['Б', 'В'], 'В': ['Б', 'П'], 
    'М': ['Н', 'Ң'], 'Н': ['М', 'Ң'], 'Ң': ['М', 'Н'], 'Г': ['Ғ'], 'Ғ': ['Г'], 
    'У': ['Ү', 'Ұ'], 'Ү': ['У', 'Ұ'], 'Ұ': ['У', 'Ү'], 'А': ['Ә'], 'Ә': ['А'], 
    'О': ['Ө'], 'Ө': ['О'], 'І': ['Ы'], 'Ы': ['І']
}

DIAGNOSTICS_DB = {
    "Қ": {
        "sub_К": "💡 Анализ: Сіздің «Қ» дыбысыңыз «К» сияқты жұмсақ естілді. Дыбысты таңдайдан емес, көмейдің (тамақтың) тереңінен жуан етіп шығаруға тырысыңыз.", 
        "sub_Х": "💡 Анализ: Сіз «Қ» дыбысын созып, «Х» сияқты айттыңыз. Тілдің түбін таңдайға қаттырақ тигізіп, кілт үзіп айтыңыз.", 
        "lips": "💡 Анализ: Аузыңыздың ашылуы жеткіліксіз. «Қ» дыбысын айтқанда ерінді еркін ұстап, ауызды кеңірек ашыңыз.", 
        "acoustic": "💡 Анализ: «Қ» дыбысы таза шықпады. Дыбысты тамақтан (көмейден) шығаруға мән беріңіз."
    },
    "Е": {
        "sub_Э": "💡 Анализ: Сіз «Е» дыбысының орнына орысша «Э» айтып қойдыңыз. Ерінді сәл жиырып, тілді алға қарай ұстап жіңішке айтыңыз.", 
        "lips": "💡 Анализ: Еріннің қалыптасуы дұрыс емес. Айнаға қарап «Е» дыбысын жаттығыңыз.", 
        "acoustic": "💡 Анализ: «Е» дыбысы дәл шықпады. Ауызыңызды тым кең ашпай, анық айтуға тырысыңыз."
    },
    "О": {
        "sub_Ө": "💡 Анализ: Сіз «О» орнына «Ө» айттыңыз. Тілді артқа тартып, жуан айту керек.", 
        "acoustic": "💡 Анализ: «О» дыбысы анық емес."
    },
    "Ө": {
        "sub_О": "💡 Анализ: Сіз «Ө» орнына жуан «О» айттыңыз. Бұл жіңішке дыбыс, тілді алға қарай ұстап айтыңыз.", 
        "acoustic": "💡 Анализ: «Ө» дыбысы таза шықпады."
    },
    "Ғ": {
        "sub_Г": "💡 Анализ: «Ғ» дыбысы «Г» сияқты естілді. Тілдің түбін артқа тартып, дыбысты тамақтан созыңқырап шығарыңыз.", 
        "sub_Қ": "💡 Анализ: Сіз «Ғ» орнына қатаң «Қ» айтып қойдыңыз. Дауыс шымылдығын дірілдетіп, ұяң айтуға тырысыңыз.", 
        "acoustic": "💡 Анализ: «Ғ» дыбысы анық емес. Көмейді пайдаланып, дірілмен айтыңыз."
    },
    "Ң": {
        "sub_Н": "💡 Анализ: Сіз «Ң» орнына «Н» айттыңыз. Тілдің ұшын тіске тіремей, тілдің түбін таңдайға көтеріңіз (мұрынмен айтылатын дыбыс).", 
        "sub_М": "💡 Анализ: Дыбыс «М» сияқты естілді. Ерінді жаппаңыз, дыбыс мұрын қуысы арқылы шығуы тиіс.", 
        "acoustic": "💡 Анализ: Мұрын жолды «Ң» дыбысы таза шықпады. Тілдің артын көтеріп жаттығыңыз."
    },
    "Ш": {
        "sub_С": "💡 Анализ: «Ш» дыбысы «С» сияқты ысқырық болып кетті. Тілдің ұшын сәл көтеріп, альвеолаға (тістің артына) жақындатыңыз.", 
        "sub_Щ": "💡 Анализ: Тым жұмсақ «Щ» естілді. Қазақтың «Ш» дыбысы әрқашан қатты айтылады.", 
        "acoustic": "💡 Анализ: Ызың дыбыс тым ақырын естілді. Демді күштірек шығарыңыз."
    },
    "П": {
        "sub_Б": "💡 Анализ: Сіз «П» орнына ұяң «Б» айттыңыз.", 
        "sub_В": "💡 Анализ: Еріндеріңіз дұрыс жұмылмады, «В» сияқты естілді.", 
        "lips": "💡 Анализ: Еріндеріңіз дұрыс жұмылмады. «П» дыбысы екі еріннің қатты соқтығысуынан шығады.", 
        "acoustic": "💡 Анализ: «П» - жарылысты дыбыс. Оны айтқанда демді кілт шығару керек."
    },
    "Ұ": {
        "sub_У": "💡 Анализ: Сіз «Ұ» орнына «У» айтып қойдыңыз. Ерінді қаттырақ дөңгелетіп, қысқа айтыңыз."
    },
    "default": {
        "lips": "💡 Анализ: Ауыздың ашылуы немесе еріннің қимылы дәл емес. Айнаға қарап, артикуляцияны дұрыстаңыз.", 
        "acoustic": "💡 Анализ: Бұл дыбыс сәл көмескі шықты. Асықпай, әр әріпті шегелеп айтуға тырысыңыз.", 
        "general": "💡 Анализ: Бұл дыбысты қайтадан айтып жаттығыңыз.", 
        "perfect": "🌟 Мінсіз! Сіздің артикуляцияңыз өте таза шықты!"
    }
}

def get_diagnostic_advice(char, err_code, score):
    if score >= 80 or err_code == "perfect":
        return DIAGNOSTICS_DB["default"]["perfect"]
        
    database_entry = DIAGNOSTICS_DB.get(char, DIAGNOSTICS_DB["default"])
    
    if str(err_code).startswith('sub_'):
        substituted_character = err_code.split('_')[1]
        if f"sub_{substituted_character}" in database_entry:
            return database_entry[f"sub_{substituted_character}"]
        else:
            return f"💡 Анализ: Сіздің «{char}» дыбысыңыз «{substituted_character}» сияқты естілді. Артикуляция орнын өзгертіп көріңіз."
            
    return database_entry.get(err_code, database_entry.get("general", DIAGNOSTICS_DB["default"]["general"]))

def get_video_frame(path):
    video_capture = cv2.VideoCapture(path)
    success, frame = video_capture.read()
    video_capture.release()
    
    if success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image)
    return None

def scan_interactive_levels(base_path="interactive_rooms"):
    interactive_levels = []
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            interactive_levels.append({
                "id": folder_name, 
                "title": format_display_name(folder_name), 
                "desc": f"{format_display_name(folder_name)} бөлмесіндегі заттар", 
                "folder": folder_path.replace("\\", "/")
            })
    return interactive_levels

def scan_live_levels(base_path="live_rooms"):
    live_levels = []
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            mission_keywords = []
            words_file_path = os.path.join(folder_path, "words.txt")

            if os.path.exists(words_file_path):
                with open(words_file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        for word in line.replace(',', ';').split(';'):
                            clean_word = word.strip().upper()
                            if clean_word and clean_word not in mission_keywords: 
                                mission_keywords.append(clean_word)
            
            if not mission_keywords: 
                mission_keywords = ["БЕЙНЕ", "АДАМДАР", "АЙНАЛА", "ТҮСТЕР", "ӘРЕКЕТ"]

            level_title = format_display_name(folder_name)
            live_levels.append({
                "id": folder_name, 
                "title": level_title, 
                "desc": f"{level_title} айналасын сипаттау", 
                "folder": folder_path.replace("\\", "/"), 
                "missions": mission_keywords
            })
    return live_levels

INTERACTIVE_LEVELS = scan_interactive_levels()
LIVE_LEVELS = scan_live_levels()

ALL_LETTERS = ["А","Ә","Б","В","Г","Ғ","Д","Е","Ё","Ж","З","И","Й","К","Қ","Л_тв","Л_мяг","М","Н","Ң","О","Ө","П","Р","С","Т","У","Ұ","Ү","Ф","Х","Һ","Ц","Ч","Ш","Щ","Ы","І","Э","Ю","Я"]
WORD_LIST = ["МЕКТЕП", "ДӘПТЕР", "ҚАЛАМ", "КІТАП", "ОҚУШЫ", "МҰҒАЛІМ", "ТАҚТА", "СЫНЫП", "САБАҚ", "БАҒА", "ОТБАСЫ", "АТА", "ӘЖЕ", "ӘКЕ", "АНА", "БАЛА", "АҒА", "ІНІ", "ҚАРЫНДАС", "СӘБИ", "ҚАЛА", "АУЫЛ", "ЖАҢБЫР", "БҰЛТ", "ЖЕЛ", "ТАУ", "ӨЗЕН", "КӨЛ", "АСПАН", "АҒАШ", "АРЫСТАН", "ЖОЛБАРЫС", "АЮ", "ҚАСҚЫР", "ТҮЛКІ", "ҚОЯН", "ПІЛ", "ЖЫЛҚЫ", "ТҮЙЕ", "СИЫР", "ШАЙ", "НАН", "СҮТ", "СУ", "АЛМА", "АЛМҰРТ", "ТЕРЕЗЕ", "ЕСІК", "ҮЙ", "ДОС", "ҚЫЛШЫҚ"]

TARGET_MOUTH_OPENNESS = {"А": 0.08, "Ә": 0.07, "О": 0.06, "Ө": 0.05, "Ұ": 0.03, "Ү": 0.02, "У": 0.04, "Ы": 0.03, "І": 0.02, "Е": 0.04, "И": 0.02, "Э": 0.05, "Қ": 0.04, "Ғ": 0.03, "Ң": 0.03, "Д": 0.02, "Т": 0.02, "В": 0.02, "Ф": 0.02, "Л_тв": 0.03, "Л_мяг": 0.02, "П": 0.0, "Б": 0.0, "М": 0.0}

def save_to_db(text, score):
    saved_data = []
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as file:
                saved_data = json.load(file)
        except Exception:
            pass
            
    saved_data.insert(0, {
        "letter": text, 
        "score": int(score), 
        "date": datetime.now().strftime("%Y-%m-%d"), 
        "time": datetime.now().strftime("%H:%M")
    })
    
    with open(DB_FILE, "w", encoding="utf-8") as file:
        json.dump(saved_data, file, ensure_ascii=False, indent=4)

def preprocess_audio_signal(audio_array):
    if len(audio_array) == 0:
        return audio_array
        
    audio_array = librosa.effects.preemphasis(audio_array)
    current_rms_volume = np.sqrt(np.mean(audio_array**2))
    
    if 0.001 < current_rms_volume < 0.5:
        audio_array = audio_array * (0.05 / current_rms_volume)
        
    return np.clip(audio_array, -1.0, 1.0)

class AudioEngine:
    def __init__(self):
        self.scaler = None
        self.classifier = None
        self.classes = []
        self.vosk_model = None
        
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, "rb") as file:
                    model_data = pickle.load(file)
                    self.scaler = model_data.get('scaler')
                    self.classifier = model_data.get('classifier')
                    if self.classifier:
                        self.classes = list(self.classifier.classes_)
            except Exception:
                pass
                
        try:
            if os.path.exists(VOSK_MODEL_PATH):
                self.vosk_model = Model(VOSK_MODEL_PATH)
        except Exception:
            pass

    def extract_features(self, audio_signal, sample_rate=16000):
        if len(audio_signal) == 0:
            return np.zeros(202)
            
        target_length = int(sample_rate * 0.3) 
        rms_energy = librosa.feature.rms(y=audio_signal)[0]
        peak_frame_index = np.argmax(rms_energy)
        peak_sample_index = librosa.frames_to_samples(peak_frame_index)
        
        start_sample = max(0, peak_sample_index - int(sample_rate * 0.1))
        end_sample = start_sample + target_length
        
        if end_sample > len(audio_signal):
            end_sample = len(audio_signal)
            start_sample = max(0, end_sample - target_length)
            
        focused_audio = audio_signal[start_sample:end_sample]
        
        if len(focused_audio) < target_length:
            focused_audio = np.pad(focused_audio, (0, target_length - len(focused_audio)))
            
        focused_audio = librosa.effects.preemphasis(focused_audio)
        focused_audio = librosa.util.normalize(focused_audio)
        
        mfcc_features = librosa.feature.mfcc(y=focused_audio, sr=sample_rate, n_mfcc=20)
        flattened_mfccs = mfcc_features.flatten() 
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(focused_audio))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=focused_audio, sr=sample_rate))
        
        return np.hstack([flattened_mfccs, zero_crossing_rate, spectral_centroid])

    def calculate_phoneme_score(self, audio_signal, target_letter):
        if not self.classifier or not self.scaler or target_letter not in self.classes:
            return 0.0, target_letter
            
        try:
            extracted_features = self.extract_features(audio_signal)
            if np.max(np.abs(extracted_features)) < 1e-6:
                return 0.0, target_letter
                
            scaled_features = self.scaler.transform([extracted_features])
            probabilities = self.classifier.predict_proba(scaled_features)[0]
            target_index = self.classes.index(target_letter)
            calculated_score = probabilities[target_index] * 100
            
            maximum_probability = np.max(probabilities) * 100 
            
            if maximum_probability > 15.0:
                top_3_indices = np.argsort(probabilities)[-3:] 
                if np.argmax(probabilities) == target_index:
                    calculated_score = max(calculated_score, 75.0 + (calculated_score * 0.2))
                elif target_index in top_3_indices:
                    calculated_score = max(calculated_score, 45.0 + (calculated_score * 0.4))

            return float(calculated_score), target_letter
        except Exception:
            return 0.0, target_letter

class TranscribeThread(QThread):
    finished = pyqtSignal(str, float, str, str)
    
    def __init__(self, speech_model, audio_array, target_word, origin_mode):
        super().__init__()
        self.speech_model = speech_model
        self.audio_array = audio_array
        self.target_word = target_word
        self.origin_mode = origin_mode
        
    def run(self):
        try:
            if self.speech_model is None:
                self.finished.emit("", 0.0, self.target_word, self.origin_mode)
                return
                
            recognizer = KaldiRecognizer(self.speech_model, 16000)
            recognizer.SetWords(True)
            
            audio_bytes = (self.audio_array * 32767).astype(np.int16).tobytes()
            recognizer.AcceptWaveform(audio_bytes)
            
            result_json = json.loads(recognizer.FinalResult())
            raw_text = result_json.get("text", "").strip().upper()
            recognized_text = "".join(character for character in raw_text if character.isalpha() or character.isspace()).replace(" ", "")
            
            confidence_list = [word_data.get("conf", 1.0) for word_data in result_json.get("result", [])]
            average_confidence = sum(confidence_list) / max(1, len(confidence_list)) if confidence_list else 0.0
            
            self.finished.emit(recognized_text, float(average_confidence), self.target_word, self.origin_mode)
        except Exception:
            self.finished.emit("", 0.0, self.target_word, self.origin_mode)

class LiveTranscribeThread(QThread):
    word_found_signal = pyqtSignal(str, float, np.ndarray)
    
    def __init__(self, speech_model, mission_keywords):
        super().__init__()
        self.speech_model = speech_model
        self.mission_keywords = mission_keywords
        self.is_active = True
        self.audio_queue = queue.Queue()
        self.full_audio_buffer = []
        self.found_mission_words = set()
        
    def add_data(self, audio_data):
        self.audio_queue.put(audio_data)
        self.full_audio_buffer.extend(audio_data)
        if len(self.full_audio_buffer) > 160000:
            self.full_audio_buffer = self.full_audio_buffer[-160000:]
            
    def run(self):
        if not self.speech_model:
            return
            
        recognizer = KaldiRecognizer(self.speech_model, 16000)
        recognizer.SetWords(True)
        
        while self.is_active:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                try:
                    audio_bytes = (np.array(audio_chunk, dtype=np.float32) * 32767).astype(np.int16).tobytes()
                    if recognizer.AcceptWaveform(audio_bytes):
                        self.process_text(json.loads(recognizer.Result()).get("text", ""))
                    else:
                        self.process_text(json.loads(recognizer.PartialResult()).get("partial", ""))
                except Exception:
                    continue
            except queue.Empty:
                continue
                
        try:
            self.process_text(json.loads(recognizer.FinalResult()).get("text", ""))
        except Exception:
            pass

    def process_text(self, text):
        if not text:
            return
            
        cleaned_text = text.upper().replace(" ", "")
        for mission in self.mission_keywords:
            if mission in self.found_mission_words:
                continue 
                
            cleaned_mission = mission.replace(" ", "")
            if cleaned_mission in cleaned_text:
                self.found_mission_words.add(mission)
                start_index = max(0, len(self.full_audio_buffer) - 80000) 
                audio_slice = np.array(self.full_audio_buffer[start_index:], dtype=np.float32)
                self.word_found_signal.emit(mission, 0.9, audio_slice)
                
    def stop(self):
        self.is_active = False
        self.wait()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, object, float, bool)
    
    def __init__(self):
        super().__init__()
        self.is_running = True
        
    def run(self):
        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
            while self.is_running:
                success, frame = video_capture.read()
                if success: 
                    frame = cv2.flip(frame, 1)
                    height, width, _ = frame.shape
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    
                    face_landmarks = results.multi_face_landmarks[0].landmark if results.multi_face_landmarks else None
                    calculated_lip_ratio = 0.0
                    is_face_aligned = False
                    
                    if face_landmarks:
                        nose_point = face_landmarks[1]
                        face_top_point = face_landmarks[10]
                        face_bottom_point = face_landmarks[152]
                        face_height = abs(face_top_point.y - face_bottom_point.y)
                        
                        is_face_aligned = (0.35 < nose_point.x < 0.65 and 0.35 < nose_point.y < 0.65 and 0.2 < face_height < 0.8)
                        
                        absolute_lip_distance = abs(face_landmarks[13].y - face_landmarks[14].y)
                        calculated_lip_ratio = absolute_lip_distance * (0.45 / max(face_height, 0.001))
                        
                    self.change_pixmap_signal.emit(frame.copy(), face_landmarks, calculated_lip_ratio, is_face_aligned)
                self.msleep(30)
        video_capture.release()
        
    def stop(self):
        self.is_running = False
        self.wait()

class BgVideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.is_running = True
        
    def run(self):
        video_capture = cv2.VideoCapture(self.video_path)
        frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
        if not frames_per_second or frames_per_second < 1:
            frames_per_second = 30
            
        frame_delay = int(1000 / frames_per_second)
        while self.is_running:
            success, frame = video_capture.read()
            if not success:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.change_pixmap_signal.emit(rgb_frame.copy())
            self.msleep(frame_delay)
            
        video_capture.release()
        
    def stop(self):
        self.is_running = False
        self.wait()

class InteractiveLetterCard(QPushButton):
    card_clicked = pyqtSignal(str, int, str) 
    
    def __init__(self, character, score, error_code):
        super().__init__()
        self.character = character
        self.score = int(score)
        self.error_code = error_code
        self.setFixedSize(70, 90)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.border_color = "#00F260" if self.score >= 80 else "#FDC830" if self.score >= 45 else "#FF4B2B"
        self.setStyleSheet(f"QPushButton {{ background: rgba(0,0,0,180); border: 2px solid {self.border_color}; border-radius: 18px; }} QPushButton:hover {{ background: rgba(40,40,50,200); transform: scale(1.1); }}")
        
        info_icon = QLabel("ℹ", self)
        info_icon.setStyleSheet("color: rgba(255,255,255,100); font-size: 12px; border: none; background: transparent;")
        info_icon.move(55, 5)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 5, 2, 5)
        main_layout.setSpacing(2)
        
        character_label = QLabel(character)
        character_label.setStyleSheet("color: white; font-weight: 900; font-size: 28px; border: none; background: transparent;")
        character_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        score_label = QLabel(f"{self.score}%")
        score_label.setStyleSheet(f"color: {self.border_color}; font-weight: bold; font-size: 16px; border: none; background: transparent;")
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        main_layout.addWidget(character_label)
        main_layout.addWidget(score_label)
        self.clicked.connect(self.on_click)
        
    def on_click(self):
        diagnostic_advice = get_diagnostic_advice(self.character, self.error_code, self.score)
        self.card_clicked.emit(self.character, self.score, diagnostic_advice)

class LiveWordCard(QPushButton):
    def __init__(self, target_word, final_score, letter_results, click_callback):
        super().__init__()
        self.target_word = target_word
        self.final_score = final_score
        self.letter_results = letter_results
        
        self.setFixedHeight(65)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("QPushButton { background: rgba(30, 30, 35, 200); border-radius: 15px; border: 1px solid rgba(255,255,255,20); text-align: left; padding: 10px; } QPushButton:hover { border-color: #F59E0B; background: rgba(40, 40, 45, 220); }")
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 0, 15, 0)
        
        word_label = QLabel(target_word)
        word_label.setStyleSheet("color: white; font-weight: 900; font-size: 18px; border: none; background: transparent;")
        
        score_color = "#00F260" if final_score >= 80 else "#FDC830" if final_score >= 45 else "#FF4B2B"
        score_label = QLabel(f"{final_score}%")
        score_label.setStyleSheet(f"color: {score_color}; font-weight: 900; font-size: 18px; border: none; background: transparent;")
        
        main_layout.addWidget(word_label)
        main_layout.addStretch()
        main_layout.addWidget(score_label)
        self.clicked.connect(lambda: click_callback(self.target_word, self.final_score, self.letter_results))

class FeedbackOverlay(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QFrame#Bg { background: rgba(0, 0, 0, 200); }")
        self.setObjectName("Bg")
        self.hide()
        
        self.dialog_box = QFrame(self)
        self.dialog_box.setStyleSheet("QFrame { background: rgba(30, 30, 35, 255); border: 2px solid rgba(255, 255, 255, 40); border-radius: 30px; }")
        
        self.dialog_layout = QVBoxLayout(self.dialog_box)
        self.dialog_layout.setContentsMargins(40, 40, 40, 40)
        self.dialog_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.character_label = QLabel("")
        self.character_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.character_label.setFixedSize(140, 140)
        self.dialog_layout.addWidget(self.character_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.dialog_layout.addSpacing(20)
        
        self.advice_label = QLabel("")
        self.advice_label.setWordWrap(True)
        self.advice_label.setFixedWidth(500)
        self.advice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dialog_layout.addWidget(self.advice_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.dialog_layout.addSpacing(30)
        
        self.close_button = QPushButton("Түсіндім")
        self.close_button.setFixedSize(220, 55)
        self.close_button.setStyleSheet("QPushButton { background: #3B82F6; color: white; border-radius: 27px; font-weight: bold; font-size: 18px; border: none; } QPushButton:hover { background: #2563EB; }")
        self.close_button.clicked.connect(self.hide_overlay)
        self.dialog_layout.addWidget(self.close_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(300)

    def mousePressEvent(self, event):
        if hasattr(self, 'dialog_box') and not self.dialog_box.geometry().contains(event.pos()):
            self.hide_overlay()
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        if hasattr(self, 'dialog_box'):
            self.dialog_box.setGeometry((self.width() - 600) // 2, (self.height() - 450) // 2, 600, 450)
        super().resizeEvent(event)

    def show_feedback(self, character, score, advice):
        border_color = "#00F260" if score >= 80 else "#FDC830" if score >= 45 else "#FF4B2B"
        if character == "!": 
            self.character_label.setText("!")
            self.character_label.setStyleSheet(f"font-size: 70px; font-weight: 900; color: #FF4B2B; background: rgba(0,0,0,100); border: 5px solid #FF4B2B; border-radius: 70px;")
        else:
            self.character_label.setText(character)
            self.character_label.setStyleSheet(f"font-size: 70px; font-weight: 900; color: white; background: rgba(0,0,0,100); border: 5px solid {border_color}; border-radius: 70px;")
            
        self.advice_label.setText(advice)
        self.advice_label.setStyleSheet("font-size: 22px; font-weight: bold; color: white; line-height: 1.5; background: transparent; border: none;")
        
        self.fade_animation.stop()
        try:
            self.fade_animation.finished.disconnect()
        except Exception:
            pass
            
        self.opacity_effect.setOpacity(0.0)
        self.raise_()
        self.show()
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.start()
        
    def hide_overlay(self):
        self.fade_animation.stop()
        self.fade_animation.setStartValue(self.opacity_effect.opacity())
        self.fade_animation.setEndValue(0.0)
        try:
            self.fade_animation.finished.disconnect()
        except Exception:
            pass
        self.fade_animation.finished.connect(self.hide)
        self.fade_animation.start()

class ToggleSwitch(QWidget):
    toggled = pyqtSignal(bool)
    
    def __init__(self, checked=False):
        super().__init__()
        self.setFixedSize(60, 32)
        self._is_checked = checked
        
        self.animation = QVariantAnimation(self)
        self.animation.setDuration(250)
        self.animation.setStartValue(2 if not checked else 28)
        self.animation.setEndValue(28 if not checked else 2)
        self.animation.valueChanged.connect(self.update_position)
        
        self._circle_position = 28 if checked else 2
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
    def update_position(self, value):
        self._circle_position = value
        self.update()
        
    def setChecked(self, state):
        self._is_checked = state
        self._circle_position = 28 if state else 2
        self.update()
        
    def isChecked(self):
        return self._is_checked
        
    def mousePressEvent(self, event):
        self._is_checked = not self._is_checked
        self.animation.setStartValue(self._circle_position)
        self.animation.setEndValue(28 if self._is_checked else 2)
        self.animation.start()
        self.toggled.emit(self._is_checked)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        background_color = QColor("#00F260") if self._is_checked else QColor("#4B5563")
        painter.setBrush(background_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 16, 16)
        
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawEllipse(self._circle_position, 2, 28, 28)

class ProgressChart(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 250)
        self.chart_data_points = [] 
        
    def set_data(self, data):
        self.chart_data_points = data
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        widget_width, widget_height = self.width(), self.height()
        
        painter.setBrush(QColor(30, 30, 35, 180))
        painter.setPen(QPen(QColor(255,255,255, 20), 1))
        painter.drawRoundedRect(0, 0, widget_width, widget_height, 20, 20)
        
        if not self.chart_data_points:
            painter.setPen(QColor("#AAA"))
            painter.setFont(QFont("Segoe UI Variable", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Бұл әріп бойынша мәлімет жоқ")
            return
            
        margin_left, margin_right, margin_top, margin_bottom = 50, 30, 30, 40
        graph_width = widget_width - margin_left - margin_right
        graph_height = widget_height - margin_top - margin_bottom
        
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1, Qt.PenStyle.DashLine))
        for i in range(5):
            y_position = margin_top + i * (graph_height / 4)
            painter.drawLine(int(margin_left), int(y_position), int(widget_width - margin_right), int(y_position))
            painter.setPen(QColor("#888"))
            painter.drawText(5, int(y_position + 5), f"{100 - i*25}%")
            painter.setPen(QPen(QColor(255, 255, 255, 30), 1, Qt.PenStyle.DashLine))
            
        data_count = len(self.chart_data_points)
        visual_points = []
        
        for i, (date_string, score) in enumerate(self.chart_data_points):
            x_position = margin_left + (i * (graph_width / max(1, data_count - 1)) if data_count > 1 else graph_width / 2)
            y_position = margin_top + graph_height - (score / 100.0) * graph_height
            visual_points.append(QPointF(float(x_position), float(y_position)))
            
            if data_count <= 7 or i % (data_count // 5 + 1) == 0 or i == data_count - 1:
                painter.setPen(QColor("#888"))
                short_date = ".".join(date_string.split("-")[1:][::-1]) if "-" in date_string else date_string
                painter.drawText(int(x_position - 15), int(widget_height - 10), short_date)
                
        if data_count > 1:
            line_path = QPainterPath()
            line_path.moveTo(visual_points[0])
            for point in visual_points[1:]:
                line_path.lineTo(point)
                
            painter.setPen(QPen(QColor("#3B82F6"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawPath(line_path)
            
            fill_path = QPainterPath(line_path)
            fill_path.lineTo(visual_points[-1].x(), margin_top + graph_height)
            fill_path.lineTo(visual_points[0].x(), margin_top + graph_height)
            fill_path.closeSubpath()
            
            gradient = QLinearGradient(0, margin_top, 0, margin_top + graph_height)
            gradient.setColorAt(0, QColor(59, 130, 246, 100))
            gradient.setColorAt(1, QColor(59, 130, 246, 0))
            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(fill_path)
            
        painter.setBrush(QColor("#00F260"))
        painter.setPen(QPen(QColor(255,255,255), 2))
        for point in visual_points:
            painter.drawEllipse(point, 6, 6)

class ProfileStatsPage(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(40, 30, 40, 30)
        
        top_bar_layout = QHBoxLayout()
        back_button = QPushButton("⬅ БАСТЫ МӘЗІР")
        back_button.setFixedSize(160, 45)
        back_button.setStyleSheet("QPushButton { background: rgba(255,255,255,0.1); color: white; border-radius: 20px; font-weight: bold; font-size: 14px; border: 1px solid rgba(255,255,255,0.2);} QPushButton:hover { background: rgba(255,255,255,0.2); }")
        back_button.clicked.connect(back_callback)
        
        page_title = QLabel("МЕНІҢ ПРОФИЛІМ ЖӘНЕ СТАТИСТИКА")
        page_title.setStyleSheet("font-size: 32px; font-weight: 900; color: white; letter-spacing: 2px;")
        
        top_bar_layout.addWidget(back_button)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(page_title)
        top_bar_layout.addStretch()
        top_bar_layout.addSpacing(160) 
        
        self.main_layout.addLayout(top_bar_layout)
        self.main_layout.addSpacing(30)
        
        content_layout = QHBoxLayout()
        
        left_column = QVBoxLayout()
        left_title = QLabel("Әріпті таңдаңыз:")
        left_title.setStyleSheet("color: #AAA; font-size: 16px; font-weight: bold;")
        left_column.addWidget(left_title)
        
        self.letters_scroll_area = QScrollArea()
        self.letters_scroll_area.setFixedWidth(120)
        self.letters_scroll_area.setWidgetResizable(True)
        self.letters_scroll_area.setStyleSheet("background: transparent; border: none;")
        
        self.letters_container_widget = QWidget()
        self.letters_layout = QVBoxLayout(self.letters_container_widget)
        self.letters_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.letters_scroll_area.setWidget(self.letters_container_widget)
        left_column.addWidget(self.letters_scroll_area)
        
        right_column = QVBoxLayout()
        self.current_letter_label = QLabel("...")
        self.current_letter_label.setStyleSheet("font-size: 48px; font-weight: 900; color: #3B82F6;")
        right_column.addWidget(self.current_letter_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.progress_chart = ProgressChart()
        right_column.addWidget(self.progress_chart)
        
        stats_horizontal_layout = QHBoxLayout()
        self.stat_card_average = self.create_stat_card("Орташа балл", "0%")
        self.stat_card_best = self.create_stat_card("Ең сәтті күн", "-")
        self.stat_card_count = self.create_stat_card("Жаттығу саны", "0")
        
        stats_horizontal_layout.addWidget(self.stat_card_average)
        stats_horizontal_layout.addWidget(self.stat_card_best)
        stats_horizontal_layout.addWidget(self.stat_card_count)
        
        right_column.addSpacing(20)
        right_column.addLayout(stats_horizontal_layout)
        
        content_layout.addLayout(left_column)
        content_layout.addSpacing(30)
        content_layout.addLayout(right_column, stretch=1)
        self.main_layout.addLayout(content_layout)

    def create_stat_card(self, title, value):
        card_frame = QFrame()
        card_frame.setStyleSheet("QFrame { background: rgba(30, 30, 35, 180); border-radius: 20px; border: 1px solid rgba(255,255,255,20); }")
        card_frame.setFixedSize(200, 100)
        
        card_layout = QVBoxLayout(card_frame)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #AAA; font-size: 14px;")
        
        value_label = QLabel(value)
        value_label.setObjectName("val")
        value_label.setStyleSheet("color: white; font-size: 28px; font-weight: bold;")
        
        card_layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(value_label, alignment=Qt.AlignmentFlag.AlignCenter)
        return card_frame

    def update_stat_card(self, card_widget, value, color="white"):
        value_label = card_widget.findChild(QLabel, "val")
        if value_label:
            value_label.setText(value)
            value_label.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: bold;")

    def load_data(self):
        for index in reversed(range(self.letters_layout.count())): 
            widget_item = self.letters_layout.itemAt(index).widget()
            if widget_item:
                widget_item.deleteLater()
            
        saved_data = []
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, "r", encoding="utf-8") as file:
                    saved_data = json.load(file)
            except Exception:
                pass
            
        self.history_dictionary = {}
        for entry in saved_data:
            letter_key = entry["letter"]
            if letter_key in ALL_LETTERS:
                if letter_key not in self.history_dictionary:
                    self.history_dictionary[letter_key] = []
                self.history_dictionary[letter_key].append(entry)
                
        if not self.history_dictionary:
            empty_label = QLabel("Деректер жоқ")
            empty_label.setStyleSheet("color: #666;")
            self.letters_layout.addWidget(empty_label)
            self.show_letter_stats("")
            return

        first_available_letter = None
        for letter_key in sorted(self.history_dictionary.keys()):
            if not first_available_letter:
                first_available_letter = letter_key
                
            letter_button = QPushButton(letter_key)
            letter_button.setFixedSize(80, 80)
            letter_button.setStyleSheet("""
                QPushButton { background: rgba(59, 130, 246, 0.2); color: white; font-size: 24px; font-weight: bold; border-radius: 20px; border: 2px solid transparent;}
                QPushButton:hover { background: rgba(59, 130, 246, 0.5); }
                QPushButton:checked { border: 2px solid #3B82F6; background: rgba(59, 130, 246, 0.8); }
            """)
            letter_button.setCheckable(True)
            letter_button.clicked.connect(lambda checked, l=letter_key: self.select_letter(l))
            self.letters_layout.addWidget(letter_button)
            
        if first_available_letter:
            self.select_letter(first_available_letter)

    def select_letter(self, selected_letter):
        for index in range(self.letters_layout.count()):
            widget_item = self.letters_layout.itemAt(index).widget()
            if isinstance(widget_item, QPushButton):
                widget_item.setChecked(widget_item.text() == selected_letter)
        self.show_letter_stats(selected_letter)

    def show_letter_stats(self, letter_key):
        self.current_letter_label.setText(letter_key if letter_key else "...")
        
        if not letter_key or letter_key not in self.history_dictionary:
            self.progress_chart.set_data([])
            self.update_stat_card(self.stat_card_average, "0%")
            self.update_stat_card(self.stat_card_best, "-")
            self.update_stat_card(self.stat_card_count, "0")
            return
            
        letter_entries = self.history_dictionary[letter_key]
        daily_scores_map = {}
        
        for entry in letter_entries:
            date_string = entry["date"]
            if date_string not in daily_scores_map:
                daily_scores_map[date_string] = []
            daily_scores_map[date_string].append(entry["score"])
            
        aggregated_chart_data = []
        for date_string in sorted(daily_scores_map.keys()):
            average_daily_score = sum(daily_scores_map[date_string]) / len(daily_scores_map[date_string])
            aggregated_chart_data.append((date_string, average_daily_score))
            
        self.progress_chart.set_data(aggregated_chart_data)
        
        all_recorded_scores = [entry["score"] for entry in letter_entries]
        total_average_score = sum(all_recorded_scores) / len(all_recorded_scores)
        average_color = "#00F260" if total_average_score >= 80 else "#FDC830" if total_average_score >= 45 else "#FF4B2B"
        
        self.update_stat_card(self.stat_card_average, f"{int(total_average_score)}%", average_color)
        self.update_stat_card(self.stat_card_count, str(len(letter_entries)))
        
        if aggregated_chart_data:
            best_date_string = max(aggregated_chart_data, key=lambda item: item[1])[0]
            short_best_date = ".".join(best_date_string.split("-")[1:][::-1]) if "-" in best_date_string else best_date_string
            self.update_stat_card(self.stat_card_best, short_best_date)

class SettingsOverlay(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QFrame#Bg { background: rgba(0, 0, 0, 180); }")
        self.setObjectName("Bg")
        self.hide()
        
        self.app_settings = load_settings()
        
        self.dialog_box = QFrame(self)
        self.dialog_box.setStyleSheet("QFrame { background: rgba(30, 30, 35, 255); border: 1px solid rgba(255, 255, 255, 30); border-radius: 25px; }")
        
        self.dialog_layout = QVBoxLayout(self.dialog_box)
        self.dialog_layout.setContentsMargins(40, 40, 40, 40)
        
        settings_title = QLabel("БАПТАУЛАР")
        settings_title.setStyleSheet("color: white; font-size: 24px; font-weight: bold; border: none; background: transparent;")
        self.dialog_layout.addWidget(settings_title, alignment=Qt.AlignmentFlag.AlignCenter)
        self.dialog_layout.addSpacing(30)
        
        status_layout = QHBoxLayout()
        svm_status_label = QLabel("🟢 SVM: OK")
        svm_status_label.setStyleSheet("color: #00F260; font-size: 14px; font-weight: bold; background: transparent; border: none;")
        
        is_vosk_available = os.path.exists(VOSK_MODEL_PATH)
        vosk_status_label = QLabel("🟢 VOSK: OK" if is_vosk_available else "🔴 VOSK: ОФФЛАЙН (Тек әріптер)")
        vosk_status_label.setStyleSheet(f"color: {'#00F260' if is_vosk_available else '#EF4444'}; font-size: 14px; font-weight: bold; background: transparent; border: none;")
        
        status_layout.addStretch()
        status_layout.addWidget(svm_status_label)
        status_layout.addSpacing(20)
        status_layout.addWidget(vosk_status_label)
        status_layout.addStretch()
        
        self.dialog_layout.addLayout(status_layout)
        self.dialog_layout.addSpacing(20)

        auto_save_layout = QHBoxLayout()
        auto_save_label = QLabel("Автоматты түрде сақтау\n(Прогрессті батырмасыз сақтау)")
        auto_save_label.setStyleSheet("color: #AAA; font-size: 16px; border: none; background: transparent;")
        
        self.auto_save_toggle_switch = ToggleSwitch(self.app_settings.get("auto_save", False))
        self.auto_save_toggle_switch.toggled.connect(self.on_auto_save_changed)
        
        auto_save_layout.addWidget(auto_save_label)
        auto_save_layout.addStretch()
        auto_save_layout.addWidget(self.auto_save_toggle_switch)
        
        self.dialog_layout.addLayout(auto_save_layout)
        self.dialog_layout.addStretch()
        
        close_button = QPushButton("ЖАБУ")
        close_button.setFixedSize(160, 45)
        close_button.setStyleSheet("QPushButton { background: rgba(255,255,255,0.1); color: white; border-radius: 22px; font-weight: bold; border: none; } QPushButton:hover { background: rgba(255,255,255,0.2); }")
        close_button.clicked.connect(self.hide)
        self.dialog_layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignCenter)

    def on_auto_save_changed(self, state):
        self.app_settings["auto_save"] = state
        save_settings(self.app_settings)

    def resizeEvent(self, event):
        if hasattr(self, 'dialog_box'):
            dialog_width, dialog_height = 450, 350
            self.dialog_box.setGeometry((self.width() - dialog_width) // 2, (self.height() - dialog_height) // 2, dialog_width, dialog_height)
        super().resizeEvent(event)

class DioramaScene(QWidget):
    def __init__(self, click_callback):
        super().__init__()
        self.click_callback = click_callback
        self.setMouseTracking(True)
        self.background_pixmap = None
        self.loaded_diorama_items = []
        self.hovered_diorama_item = None
        
    def load_location(self, folder_path):
        self.loaded_diorama_items.clear()
        self.background_pixmap = None
        self.hovered_diorama_item = None
        
        if not os.path.exists(folder_path): 
            return
            
        folder_name = os.path.basename(folder_path)
        background_image_path = os.path.join(folder_path, f"{folder_name}.jpg")
        
        if os.path.exists(background_image_path): 
            self.background_pixmap = QPixmap(background_image_path)
        elif os.path.exists(os.path.join(folder_path, "bg.jpg")): 
            self.background_pixmap = QPixmap(os.path.join(folder_path, "bg.jpg"))
            
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(".png"):
                item_word = os.path.splitext(file_name)[0].upper().split('(')[0].strip() 
                self.loaded_diorama_items.append({
                    "word": item_word, 
                    "pixmap": QPixmap(os.path.join(folder_path, file_name)), 
                    "image": QPixmap(os.path.join(folder_path, file_name)).toImage(), 
                    "scaled_pixmap": None, 
                    "scaled_image": None
                })
                
        self.resizeEvent(None)
        self.update()
        
    def resizeEvent(self, event):
        if self.background_pixmap: 
            self.scaled_background = self.background_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            
        for item in self.loaded_diorama_items:
            item["scaled_pixmap"] = item["pixmap"].scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            item["scaled_image"] = item["scaled_pixmap"].toImage()
            
    def mouseMoveEvent(self, event):
        found_item = None
        for item in reversed(self.loaded_diorama_items):
            if item["scaled_image"] and event.pos().x() < item["scaled_image"].width() and event.pos().y() < item["scaled_image"].height():
                if item["scaled_image"].pixelColor(event.pos().x(), event.pos().y()).alpha() > 20: 
                    found_item = item
                    break
                    
        if found_item != self.hovered_diorama_item: 
            self.hovered_diorama_item = found_item
            self.setCursor(Qt.CursorShape.PointingHandCursor if found_item else Qt.CursorShape.ArrowCursor)
            self.update() 
            
    def mousePressEvent(self, event):
        if self.hovered_diorama_item and event.button() == Qt.MouseButton.LeftButton: 
            self.click_callback(self.hovered_diorama_item["word"])
            
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        if not self.background_pixmap: 
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Сурет табылмады (Фонды қосыңыз)")
            return
            
        painter.drawPixmap(0, 0, self.scaled_background)
        
        if self.hovered_diorama_item: 
            painter.fillRect(self.rect(), QColor(0, 0, 0, 70)) 
            
        for item in self.loaded_diorama_items:
            if item == self.hovered_diorama_item:
                painter.setOpacity(0.3)
                for offset_x, offset_y in [(-4,-4), (4,-4), (-4,4), (4,4), (0,-6), (0,6), (-6,0), (6,0)]: 
                    painter.drawPixmap(offset_x, offset_y, item["scaled_pixmap"])
                painter.setOpacity(1.0)
                painter.drawPixmap(0, 0, item["scaled_pixmap"])
            elif not self.hovered_diorama_item: 
                painter.drawPixmap(0, 0, item["scaled_pixmap"])


class MenuHoverCard(QPushButton):
    def __init__(self, icon_text, title_text, subtitle_text, gradient_color_1, gradient_color_2):
        super().__init__()
        self.gradient_color_1 = gradient_color_1
        self.gradient_color_2 = gradient_color_2
        self.setFixedSize(300, 180)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        card_layout = QVBoxLayout(self)
        card_layout.setContentsMargins(20, 20, 20, 20)
        
        self.icon_label = QLabel(icon_text)
        self.icon_label.setStyleSheet("font-size: 42px; font-weight: 900; color: white; background: transparent; border: none;")
        
        self.title_label = QLabel(title_text)
        self.title_label.setStyleSheet("font-size: 22px; font-weight: 900; color: white; background: transparent; border: none; margin-top: 5px;")
        
        self.subtitle_label = QLabel(subtitle_text)
        self.subtitle_label.setStyleSheet("font-size: 13px; font-weight: bold; color: rgba(255,255,255,180); background: transparent; border: none;")
        self.subtitle_label.setWordWrap(True)
        
        card_layout.addWidget(self.icon_label)
        card_layout.addStretch()
        card_layout.addWidget(self.title_label)
        card_layout.addWidget(self.subtitle_label)
        
        self.hover_animation = QVariantAnimation(self)
        self.hover_animation.setDuration(200)
        self.hover_animation.valueChanged.connect(self.update_scale_factor)
        self.current_scale_factor = 1.0
        
    def update_scale_factor(self, value):
        self.current_scale_factor = value
        self.update()
        
    def enterEvent(self, event):
        self.hover_animation.setStartValue(self.current_scale_factor)
        self.hover_animation.setEndValue(1.05)
        self.hover_animation.start()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.hover_animation.setStartValue(self.current_scale_factor)
        self.hover_animation.setEndValue(1.0)
        self.hover_animation.start()
        super().leaveEvent(event)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        widget_width, widget_height = self.width(), self.height()
        painter.translate(widget_width/2, widget_height/2)
        painter.scale(self.current_scale_factor, self.current_scale_factor)
        painter.translate(-widget_width/2, -widget_height/2)
        
        drawing_rectangle = QRectF(0, 0, widget_width, widget_height)
        linear_gradient = QLinearGradient(0, 0, widget_width, widget_height)
        linear_gradient.setColorAt(0, QColor(self.gradient_color_1))
        linear_gradient.setColorAt(1, QColor(self.gradient_color_2))
        
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(linear_gradient)
        painter.drawRoundedRect(drawing_rectangle, 25, 25)
        
        if self.current_scale_factor > 1.0:
            painter.setPen(QPen(QColor(255, 255, 255, 60), 2))
            painter.drawRoundedRect(drawing_rectangle.adjusted(1,1,-1,-1), 25, 25)

class PulseMicButton(QPushButton):
    def __init__(self):
        super().__init__()
        self.setFixedSize(100, 100)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.pulse_circle_radius = 0.0
        self.is_currently_processing = False
        
        self.pulse_animation = QVariantAnimation(self)
        self.pulse_animation.setDuration(1200)
        self.pulse_animation.setStartValue(0.0)
        self.pulse_animation.setEndValue(30.0)
        self.pulse_animation.valueChanged.connect(self.update_pulse_radius)
        self.pulse_animation.setLoopCount(-1)
        
        self.scale_animation = QVariantAnimation(self)
        self.scale_animation.setDuration(150)
        self.scale_animation.valueChanged.connect(self.update_scale_factor)
        self.current_scale_factor = 1.0
        
    def update_pulse_radius(self, value):
        self.pulse_circle_radius = value
        self.update()
        
    def update_scale_factor(self, value):
        self.current_scale_factor = value
        self.update()
        
    def enterEvent(self, event):
        if not self.is_currently_processing:
            self.scale_animation.setStartValue(self.current_scale_factor)
            self.scale_animation.setEndValue(1.1)
            self.scale_animation.start()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.scale_animation.setStartValue(self.current_scale_factor)
        self.scale_animation.setEndValue(1.0)
        self.scale_animation.start()
        super().leaveEvent(event)
        
    def set_processing(self, state):
        self.is_currently_processing = state
        self.update()
        
    def nextCheckState(self):
        super().nextCheckState()
        if self.isChecked():
            self.pulse_animation.start()
        else:
            self.pulse_animation.stop()
            self.pulse_circle_radius = 0.0
            self.update()
            
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center_point = self.rect().center()
        
        widget_width, widget_height = self.width(), self.height()
        painter.translate(widget_width/2, widget_height/2)
        painter.scale(self.current_scale_factor, self.current_scale_factor)
        painter.translate(-widget_width/2, -widget_height/2)
        
        if self.isChecked() and not self.is_currently_processing:
            painter.setPen(Qt.PenStyle.NoPen)
            alpha_channel = max(0, int(255 - (self.pulse_circle_radius / 30.0) * 255))
            painter.setBrush(QColor(239, 68, 68, alpha_channel))
            painter.drawEllipse(center_point, int(35 + self.pulse_circle_radius), int(35 + self.pulse_circle_radius))
            
        button_color = QColor("#6B7280") if self.is_currently_processing else (QColor("#EF4444") if self.isChecked() else QColor("#3B82F6"))
        button_icon_text = "⏳" if self.is_currently_processing else "🎙"
        
        painter.setBrush(button_color)
        painter.setPen(QPen(QColor(255,255,255,40), 4))
        painter.drawEllipse(center_point, 35, 35)
        
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Segoe UI Emoji", 24))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, button_icon_text)

class CircularScore(QWidget):
    def __init__(self, parent=None): 
        super().__init__(parent)
        self.current_display_value = 0
        self.setMinimumSize(140, 140)
        self.setCursor(Qt.CursorShape.PointingHandCursor) 
        
    def set_score(self, value):
        self.current_display_value = value
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        circle_size = min(self.width(), self.height()) - 20
        drawing_rectangle = QRectF((self.width() - circle_size) / 2, (self.height() - circle_size) / 2, circle_size, circle_size)
        
        painter.setPen(QPen(QColor(255, 255, 255, 15), 14))
        painter.drawEllipse(drawing_rectangle)
        
        score_color = QColor("#00F260") if self.current_display_value >= 80 else QColor("#FDC830") if self.current_display_value >= 45 else QColor("#FF4B2B")
        
        painter.setPen(QPen(QColor(score_color.red(), score_color.green(), score_color.blue(), 50), 20, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(drawing_rectangle, 90 * 16, -int(self.current_display_value * 3.6 * 16))
        
        painter.setPen(QPen(score_color, 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(drawing_rectangle, 90 * 16, -int(self.current_display_value * 3.6 * 16))
        
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Segoe UI Variable", int(circle_size / 3.5), QFont.Weight.Bold))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{int(self.current_display_value)}%")

class VoiceWave(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(40)
        self.volume_samples = [0] * 30
        
    def update_wave(self, volume_level):
        self.volume_samples.pop(0)
        self.volume_samples.append(volume_level)
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar_width = self.width() / len(self.volume_samples)
        
        for index, sample_value in enumerate(self.volume_samples):
            bar_height = max(4, sample_value * self.height() * 0.9)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(59, 130, 246, 200))
            painter.drawRoundedRect(QRectF(index * bar_width + 2, (self.height() - bar_height) / 2, bar_width - 4, bar_height), 2, 2)

class KaspiOvalOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.is_face_aligned = False
        
    def set_aligned(self, aligned_state):
        if self.is_face_aligned != aligned_state:
            self.is_face_aligned = aligned_state
            self.update()
            
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        widget_width, widget_height = self.width(), self.height()
        center_x, center_y = widget_width / 2, widget_height / 2
        overlay_path = QPainterPath()
        overlay_path.addRect(QRectF(0, 0, widget_width, widget_height)) 
        
        radius_x, radius_y = 240, 340  
        oval_rectangle = QRectF(center_x - radius_x, center_y - radius_y, radius_x * 2, radius_y * 2)
        oval_cutout_path = QPainterPath()
        oval_cutout_path.addEllipse(oval_rectangle)
        overlay_path = overlay_path.subtracted(oval_cutout_path)
        
        painter.fillPath(overlay_path, QColor(0, 0, 0, 0)) 
        
        if self.is_face_aligned:
            main_outline_color = QColor(0, 242, 96, 255)
            glow_outline_color = QColor(0, 242, 96, 80)
        else:
            main_outline_color = QColor(255, 255, 255, 200)
            glow_outline_color = QColor(255, 255, 255, 30)
            
        painter.setPen(QPen(glow_outline_color, 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawEllipse(oval_rectangle)
        
        gradient = QLinearGradient(oval_rectangle.topLeft(), oval_rectangle.bottomRight())
        if self.is_face_aligned:
            gradient.setColorAt(0, QColor(100, 255, 150, 255))
            gradient.setColorAt(1, QColor(0, 180, 50, 255))
        else:
            gradient.setColorAt(0, QColor(255, 255, 255, 255))
            gradient.setColorAt(1, QColor(150, 150, 150, 150))
            
        painter.setPen(QPen(gradient, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawEllipse(oval_rectangle)
        
        arc_length = 20 * 16 
        painter.setPen(QPen(main_outline_color, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(oval_rectangle.toRect(), 35 * 16, arc_length)
        painter.drawArc(oval_rectangle.toRect(), 125 * 16, arc_length)
        painter.drawArc(oval_rectangle.toRect(), 215 * 16, arc_length)
        painter.drawArc(oval_rectangle.toRect(), 305 * 16, arc_length)  
        
        painter.setFont(QFont("Segoe UI Variable", 16, QFont.Weight.Bold))
        instruction_text = "Бет-әлпетіңізді овалға туралаңыз" if not self.is_face_aligned else "Тамаша! Бет-әлпетіңіз анықталды"
        
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(QRectF(1, center_y - radius_y - 60 + 1, widget_width, 40), Qt.AlignmentFlag.AlignCenter, instruction_text)
        painter.setPen(QColor(255, 255, 255) if not self.is_face_aligned else QColor(0, 242, 96))
        painter.drawText(QRectF(0, center_y - radius_y - 60, widget_width, 40), Qt.AlignmentFlag.AlignCenter, instruction_text)

class SoyleAI(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(1150, 750) 
        self.setWindowTitle("SöyleAI")
        
        self.is_recording_audio = False
        self.is_diorama_mode_recording = False
        self.is_live_mode_recording = False
        self.is_camera_hidden = False
        
        self.audio_capture_data = []
        self.lip_activity_history = []
        self.current_volume_level = 0
        self.last_calculated_final_score = 0
        
        self.current_app_mode = "letter"
        self.current_target_word_to_save = ""
        self.recognized_word_html_display = ""
        
        self.word_transcription_thread = None
        self.live_transcription_thread = None
        self.current_live_level_data = None
        
        self.audio_engine = AudioEngine()
        
        self.video_processing_thread = VideoThread()
        self.video_processing_thread.change_pixmap_signal.connect(self.process_camera_frame)
        self.video_processing_thread.start()
        
        self.background_video_thread = None
        
        self.init_user_interface()
        
        self.audio_wave_timer = QTimer()
        self.audio_wave_timer.timeout.connect(lambda: self.voice_wave_display.update_wave(self.current_volume_level if (self.is_recording_audio or self.is_diorama_mode_recording or self.is_live_mode_recording) else 0))
        self.audio_wave_timer.start(30)

    @staticmethod
    def apply_soft_boost(score):
        ratio = score / 100.0
        if 0.15 <= ratio <= 0.95:
            ratio = ratio + 0.18 * np.sin(np.pi * (ratio - 0.15) / 0.80)
        return min(100.0, max(0.0, ratio * 100.0))

    def init_user_interface(self):
        self.setStyleSheet("QWidget { background-color: #0A0A0C; color: #F0F0F0; font-family: 'Segoe UI Variable', sans-serif; }")
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.view_stack = QStackedWidget()
        
        main_menu_widget = QWidget()
        menu_layout = QVBoxLayout(main_menu_widget)
        menu_layout.setContentsMargins(40, 40, 40, 40) 
        
        dashboard_top_bar = QHBoxLayout()
        dashboard_top_bar.addStretch()
        
        statistics_button = QPushButton("📊 СТАТИСТИКА")
        statistics_button.setFixedSize(160, 40)
        statistics_button.setStyleSheet("QPushButton { background: rgba(59, 130, 246, 0.2); color: #3B82F6; border-radius: 20px; font-weight: bold; border: 1px solid #3B82F6; } QPushButton:hover { background: rgba(59, 130, 246, 0.4); }")
        statistics_button.clicked.connect(self.show_statistics_page)
        
        settings_button = QPushButton("⚙️ БАПТАУЛАР")
        settings_button.setFixedSize(160, 40)
        settings_button.setStyleSheet("QPushButton { background: rgba(255, 255, 255, 0.1); color: white; border-radius: 20px; font-weight: bold; border: 1px solid rgba(255,255,255,0.3); } QPushButton:hover { background: rgba(255, 255, 255, 0.2); }")
        settings_button.clicked.connect(self.show_settings_overlay)
        
        dashboard_top_bar.addWidget(statistics_button)
        dashboard_top_bar.addSpacing(10)
        dashboard_top_bar.addWidget(settings_button)
        menu_layout.addLayout(dashboard_top_bar)
        
        title_container_layout = QVBoxLayout()
        app_title = QLabel("SöyleAI")
        app_title.setStyleSheet("font-size: 60px; font-weight: 900; color: white; letter-spacing: 2px; margin-top: 20px;") 
        
        app_subtitle = QLabel("Интеллектуалды логопед және тіл үйрету жүйесі")
        app_subtitle.setStyleSheet("font-size: 18px; font-weight: bold; color: #3B82F6;")
        
        title_container_layout.addWidget(app_title, alignment=Qt.AlignmentFlag.AlignCenter)
        title_container_layout.addWidget(app_subtitle, alignment=Qt.AlignmentFlag.AlignCenter)
        menu_layout.addLayout(title_container_layout)
        menu_layout.addStretch(1)
        
        menu_grid_layout = QGridLayout()
        menu_grid_layout.setSpacing(20)
        menu_grid_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        card_letters_mode = MenuHoverCard("🔤", "ӘРІПТЕР", "Дыбыстарды дұрыс артикуляциямен айтуға машықтану", "#1D4ED8", "#3B82F6")
        card_letters_mode.clicked.connect(lambda: self.set_application_mode("letter"))
        menu_grid_layout.addWidget(card_letters_mode, 0, 0)
        
        card_words_mode = MenuHoverCard("📝", "СӨЗДЕР", "Толық сөздерді таза әрі анық айтып үйрену", "#6D28D9", "#8B5CF6")
        card_words_mode.clicked.connect(lambda: self.set_application_mode("word"))
        menu_grid_layout.addWidget(card_words_mode, 0, 1)
        
        card_interactive_mode = MenuHoverCard("🛋️", "ИНТЕРАКТИВ", "Бөлмедегі заттарды тауып, атауларын дұрыс айту", "#047857", "#10B981")
        card_interactive_mode.clicked.connect(lambda: self.view_stack.setCurrentIndex(2))
        menu_grid_layout.addWidget(card_interactive_mode, 1, 0)
        
        card_live_mode = MenuHoverCard("🎥", "СИПАТТАМА", "Табиғат пен қоршаған ортаны жанды дауыспен сипаттау", "#B45309", "#F59E0B")
        card_live_mode.clicked.connect(lambda: self.view_stack.setCurrentIndex(4))
        menu_grid_layout.addWidget(card_live_mode, 1, 1)
        
        menu_layout.addLayout(menu_grid_layout)
        menu_layout.addStretch(1)

        self.settings_overlay_widget = SettingsOverlay(self)

        self.training_page_widget = QWidget()
        
        self.video_feed_label = QLabel(self.training_page_widget)
        self.video_feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_feed_label.setStyleSheet("background: #000;")
        
        self.oval_face_overlay = KaspiOvalOverlay(self.video_feed_label)
        
        PILL_CONTAINER_STYLE = "QFrame { background: rgba(20, 20, 25, 180); border: 1px solid rgba(255, 255, 255, 30); border-radius: 20px; }"
        
        self.back_to_menu_button = QPushButton("⬅ МӘЗІР", self.training_page_widget)
        self.back_to_menu_button.setFixedSize(120, 45)
        self.back_to_menu_button.setStyleSheet("QPushButton { background: rgba(20, 20, 25, 180); border: 1px solid rgba(255,255,255,30); color: white; border-radius: 20px; font-weight: bold;} QPushButton:hover{background: rgba(255,255,255,0.2);}")
        self.back_to_menu_button.clicked.connect(self.exit_training_page)
        
        self.center_selection_pill = QFrame(self.training_page_widget)
        self.center_selection_pill.setStyleSheet(PILL_CONTAINER_STYLE)
        selection_pill_layout = QHBoxLayout(self.center_selection_pill)
        selection_pill_layout.setContentsMargins(15, 5, 15, 5)
        
        self.selection_title_label = QLabel("ТАҢДАЛҒАН ӘРІП:")
        self.selection_title_label.setStyleSheet("color: #AAA; font-weight: bold; background: transparent; border:none;")
        
        COMBO_BOX_STYLE = "QComboBox { background: transparent; padding: 5px; font-size: 20px; font-weight: bold; color: white; border: none;} QComboBox::drop-down { border: none; } QComboBox QAbstractItemView { min-width: 180px; }"
        
        self.letter_combo_box = QComboBox()
        self.letter_combo_box.setMinimumWidth(80)
        self.letter_combo_box.addItems(ALL_LETTERS)
        self.letter_combo_box.setStyleSheet(COMBO_BOX_STYLE)
        self.letter_combo_box.currentIndexChanged.connect(self.on_letter_selection_changed)
        
        self.word_combo_box = QComboBox()
        self.word_combo_box.setMinimumWidth(200) 
        self.word_combo_box.addItems(WORD_LIST)
        self.word_combo_box.setStyleSheet(COMBO_BOX_STYLE)
        self.word_combo_box.view().setMinimumWidth(200) 
        self.word_combo_box.hide()
        self.word_combo_box.currentIndexChanged.connect(self.on_word_selection_changed)
        
        selection_pill_layout.addWidget(self.selection_title_label)
        selection_pill_layout.addWidget(self.letter_combo_box)
        selection_pill_layout.addWidget(self.word_combo_box)
        
        self.camera_toggle_button = QPushButton("📷", self.training_page_widget)
        self.camera_toggle_button.setFixedSize(45, 45)
        self.camera_toggle_button.setStyleSheet("QPushButton { background: rgba(20, 20, 25, 180); border: 1px solid rgba(255,255,255,30); color: white; border-radius: 22px; font-size: 18px;} QPushButton:hover{background: rgba(255,255,255,0.2);}")
        self.camera_toggle_button.clicked.connect(self.toggle_camera_visibility)
        
        self.score_display_pill = QFrame(self.training_page_widget)
        self.score_display_pill.setStyleSheet(PILL_CONTAINER_STYLE)
        score_pill_layout = QVBoxLayout(self.score_display_pill)
        score_pill_layout.setContentsMargins(15,15,15,15)
        
        self.circular_score_widget = CircularScore()
        self.circular_score_widget.setFixedSize(120, 120)
        self.circular_score_widget.mousePressEvent = self.on_score_widget_clicked 
        score_pill_layout.addWidget(self.circular_score_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        
        score_pill_layout.addSpacing(5) 
        self.score_hint_label = QLabel("ℹ Қатені көру")
        self.score_hint_label.setStyleSheet("color: #AAA; font-size: 11px; font-weight: bold; background: transparent; border: none;")
        self.score_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_pill_layout.addWidget(self.score_hint_label)
        
        self.main_save_result_button = QPushButton("НӘТИЖЕНІ САҚТАУ ✓", self.training_page_widget)
        self.main_save_result_button.setFixedSize(240, 60)
        self.main_save_result_button.setStyleSheet("QPushButton { background: #00F260; color: black; border-radius: 30px; font-weight: bold; font-size: 18px; border:none;} QPushButton:hover{background: #00D250;} QPushButton:disabled { background: #555; color: #888; }")
        self.main_save_result_button.hide()
        self.main_save_result_button.clicked.connect(self.handle_save_result_click)
        
        self.voice_wave_display = VoiceWave()
        self.voice_wave_display.setParent(self.training_page_widget)
        self.voice_wave_display.setFixedSize(240, 40)
        self.voice_wave_display.setStyleSheet("background: transparent; border: none;")
        
        self.record_audio_button = PulseMicButton()
        self.record_audio_button.setParent(self.training_page_widget)
        self.record_audio_button.clicked.connect(self.toggle_audio_recording)
        
        self.word_breakdown_container_widget = QWidget(self.training_page_widget)
        self.word_breakdown_layout = QHBoxLayout(self.word_breakdown_container_widget)
        self.word_breakdown_layout.setSpacing(8)
        self.word_breakdown_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.word_breakdown_container_widget.setStyleSheet("background: transparent; border: none;")
        self.word_breakdown_container_widget.hide()

        self.word_mode_hint_label = QLabel("ℹ️ Қатені көру үшін әріпті басыңыз", self.training_page_widget)
        self.word_mode_hint_label.setStyleSheet("color: rgba(255,255,255,180); font-size: 14px; background: rgba(0,0,0,100); border-radius: 10px; padding: 5px 15px; font-weight: bold;")
        self.word_mode_hint_label.hide()

        self.training_feedback_overlay = FeedbackOverlay(self.training_page_widget)

        def create_level_selector_page(levels_data_list, target_stack_index, theme_color):
            selector_page = QWidget()
            page_layout = QVBoxLayout(selector_page)
            page_layout.setContentsMargins(50, 20, 50, 30)
            
            top_bar_layout = QHBoxLayout()
            back_to_menu_btn = QPushButton("⬅ МӘЗІР")
            back_to_menu_btn.setFixedSize(140, 50)
            back_to_menu_btn.setStyleSheet("QPushButton { background: #252525; color: white; border-radius: 25px; font-weight: bold; font-size: 16px;} QPushButton:hover { background: #333333; }")
            back_to_menu_btn.clicked.connect(lambda: self.view_stack.setCurrentIndex(0))
            top_bar_layout.addWidget(back_to_menu_btn)
            top_bar_layout.addStretch()
            page_layout.addLayout(top_bar_layout)
            page_layout.addSpacing(10)
            
            page_title_label = QLabel("ИНТЕРАКТИВТІ БӨЛМЕЛЕР" if target_stack_index == 3 else "ЖАНДЫ СИПАТТАМА")
            page_title_label.setStyleSheet("font-size: 42px; font-weight: 900; color: white; letter-spacing: 2px;")
            page_layout.addWidget(page_title_label, alignment=Qt.AlignmentFlag.AlignCenter)
            page_layout.addSpacing(15)
            
            cards_scroll_area = QScrollArea()
            cards_scroll_area.setFixedHeight(380) 
            cards_scroll_area.setWidgetResizable(True)
            cards_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
            cards_scroll_area.setStyleSheet("background: transparent;")
            cards_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            cards_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            
            cards_container_widget = QWidget()
            cards_container_widget.setStyleSheet("background: transparent;")
            cards_container_layout = QHBoxLayout(cards_container_widget)
            cards_container_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            cards_container_layout.setSpacing(40)
            cards_container_layout.setContentsMargins(10, 10, 10, 10)
            
            for level_data in levels_data_list:
                level_card = QFrame()
                level_card.setFixedSize(280, 340) 
                level_card.setStyleSheet(f"QFrame {{ background: #161618; border-radius: 30px; border: 2px solid #2C2C2E; }} QFrame:hover {{ border: 2px solid {theme_color}; background: #1C1C1E; }}")
                card_inner_layout = QVBoxLayout(level_card)
                card_inner_layout.setContentsMargins(15, 15, 15, 15)
                
                cover_image_label = QLabel()
                folder_name = os.path.basename(level_data["folder"])
                image_path = os.path.join(level_data["folder"], f"{folder_name}.jpg")
                
                if not os.path.exists(image_path): 
                    image_path = os.path.join(level_data["folder"], "cover.jpg")
                if not os.path.exists(image_path): 
                    image_path = os.path.join(level_data["folder"], "bg.jpg")
                    
                if os.path.exists(image_path): 
                    cover_image_label.setPixmap(QPixmap(image_path).scaled(240, 140, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation))
                else: 
                    cover_image_label.setText("Сурет жоқ")
                    cover_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    cover_image_label.setStyleSheet("color: #666; font-size: 16px; border: 2px dashed #444; border-radius: 20px;")
                    
                cover_image_label.setFixedSize(240, 140)
                cover_image_label.setStyleSheet("background: transparent; border: none;")
                card_inner_layout.addWidget(cover_image_label, alignment=Qt.AlignmentFlag.AlignCenter)
                
                level_title_label = QLabel(level_data["title"])
                level_title_label.setStyleSheet("font-size: 24px; font-weight: 900; color: white; border: none; background: transparent; margin-top: 10px;")
                card_inner_layout.addWidget(level_title_label, alignment=Qt.AlignmentFlag.AlignCenter)
                
                level_description_label = QLabel(level_data["desc"])
                level_description_label.setStyleSheet("font-size: 12px; color: #AAA; border:none; background: transparent;")
                level_description_label.setWordWrap(True)
                level_description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                card_inner_layout.addWidget(level_description_label)
                
                enter_level_button = QPushButton("КІРУ")
                enter_level_button.setFixedHeight(45)
                enter_level_button.setStyleSheet(f"QPushButton {{ background: {theme_color}; color: white; border-radius: 15px; font-size: 15px; font-weight: bold; border: none;}} QPushButton:hover{{ opacity: 0.9; }}")
                enter_level_button.clicked.connect(lambda _, l_data=level_data: self.open_diorama_level(l_data) if target_stack_index == 3 else self.open_live_level(l_data))
                
                card_inner_layout.addStretch()
                card_inner_layout.addWidget(enter_level_button)
                cards_container_layout.addWidget(level_card)
                
            cards_scroll_area.setWidget(cards_container_widget)
            page_layout.addWidget(cards_scroll_area)
            return selector_page

        self.interactive_level_selector_page = create_level_selector_page(INTERACTIVE_LEVELS, 3, "#10B981")
        self.live_level_selector_page = create_level_selector_page(LIVE_LEVELS, 5, "#F59E0B")

        self.diorama_view_page = QWidget()
        diorama_layout = QVBoxLayout(self.diorama_view_page)
        diorama_layout.setContentsMargins(0, 0, 0, 0)
        self.diorama_interactive_scene = DioramaScene(self.open_diorama_overlay)
        diorama_layout.addWidget(self.diorama_interactive_scene)
        
        back_from_diorama_button = QPushButton("⬅ МӘЗІРГЕ ҚАЙТУ", self.diorama_view_page)
        back_from_diorama_button.setGeometry(40, 40, 220, 60)
        back_from_diorama_button.setStyleSheet("QPushButton { background: rgba(20, 20, 25, 200); border: 2px solid rgba(255,255,255,40); border-radius: 30px; color: white; font-weight:bold; font-size: 16px;} QPushButton:hover { background: #EF4444; border-color: #EF4444; }")
        back_from_diorama_button.clicked.connect(self.exit_diorama_mode)
        
        self.diorama_action_overlay = QFrame(self.diorama_view_page)
        self.diorama_action_overlay.setStyleSheet("QFrame { background: rgba(20, 20, 25, 240); border-radius: 40px; border: 2px solid #3B82F6; }")
        self.diorama_action_overlay.hide()
        
        overlay_inner_layout = QVBoxLayout(self.diorama_action_overlay)
        overlay_inner_layout.setContentsMargins(40, 40, 40, 40)
        
        top_overlay_bar = QHBoxLayout()
        
        self.diorama_target_prefix_label = QLabel("СӨЗ: ")
        self.diorama_target_prefix_label.setStyleSheet("font-size: 42px; font-weight: 900; color: white; background: transparent; border: none;")
        
        self.diorama_target_word_label = QLabel("")
        self.diorama_target_word_label.setStyleSheet("font-size: 42px; font-weight: 900; color: white; background: transparent; border: none;")
        
        self.diorama_word_blur_effect = QGraphicsBlurEffect()
        self.diorama_word_blur_effect.setBlurRadius(80) 
        self.diorama_target_word_label.setGraphicsEffect(self.diorama_word_blur_effect)
        
        self.diorama_hint_button = QPushButton("Көмек 💡")
        self.diorama_hint_button.setFixedSize(140, 45)
        self.diorama_hint_button.setStyleSheet("QPushButton { background: #F59E0B; border-radius: 22px; color: white; font-weight: bold; font-size: 16px; border: none; } QPushButton:hover { background: #D97706; }")
        self.diorama_hint_button.clicked.connect(self.reveal_diorama_hidden_word)

        close_overlay_button = QPushButton("✖")
        close_overlay_button.setFixedSize(50, 50)
        close_overlay_button.setStyleSheet("QPushButton { background: #EF4444; border-radius: 25px; color: white; font-weight: bold; font-size: 20px; border: none;}")
        close_overlay_button.clicked.connect(self.close_diorama_overlay_window)
        
        top_overlay_bar.addWidget(self.diorama_target_prefix_label)
        top_overlay_bar.addWidget(self.diorama_target_word_label)
        top_overlay_bar.addSpacing(20)
        top_overlay_bar.addWidget(self.diorama_hint_button)
        top_overlay_bar.addStretch()
        top_overlay_bar.addWidget(close_overlay_button)
        overlay_inner_layout.addLayout(top_overlay_bar)
        
        self.diorama_overlay_view_stack = QStackedWidget()
        self.diorama_overlay_view_stack.setStyleSheet("background: transparent; border: none;")
        
        start_recording_page = QWidget()
        start_page_layout = QVBoxLayout(start_recording_page)
        start_page_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.start_diorama_recording_button = QPushButton("🎙 БАСТАУ")
        self.start_diorama_recording_button.setFixedSize(300, 90)
        self.start_diorama_recording_button.setStyleSheet("QPushButton { background: #3B82F6; border-radius: 45px; color: white; font-size: 24px; font-weight: bold; border:none;}")
        self.start_diorama_recording_button.clicked.connect(self.start_diorama_audio_recording)
        start_page_layout.addWidget(self.start_diorama_recording_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        active_recording_page = QWidget()
        active_page_layout = QVBoxLayout(active_recording_page)
        active_page_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diorama_recording_status_label = QLabel("Тыңдап тұрмын...")
        self.diorama_recording_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diorama_recording_status_label.setStyleSheet("color: #00F260; font-size: 26px; font-weight: bold; border:none; margin-bottom: 30px;")
        self.stop_diorama_recording_button = QPushButton("ТОҚТАТУ")
        self.stop_diorama_recording_button.setFixedSize(300, 90)
        self.stop_diorama_recording_button.setStyleSheet("QPushButton { background: #EF4444; border-radius: 45px; color: white; font-size: 24px; font-weight: bold; border:none;}")
        self.stop_diorama_recording_button.clicked.connect(self.stop_diorama_audio_recording)
        active_page_layout.addWidget(self.diorama_recording_status_label)
        active_page_layout.addWidget(self.stop_diorama_recording_button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        results_page = QWidget()
        results_page_layout = QVBoxLayout(results_page)
        results_page_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.diorama_result_subtitle_label = QLabel("")
        self.diorama_result_subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diorama_result_subtitle_label.setStyleSheet("color: #AAA; font-size: 18px; border: none; margin-bottom: 10px;")
        
        self.diorama_letters_breakdown_layout = QHBoxLayout()
        self.diorama_letters_breakdown_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diorama_letters_breakdown_layout.setSpacing(10)
        
        self.diorama_result_hint_label = QLabel("ℹ️ Қатені көру үшін әріпті басыңыз")
        self.diorama_result_hint_label.setStyleSheet("color: rgba(255,255,255,180); font-size: 14px; background: rgba(0,0,0,100); border-radius: 10px; padding: 5px 15px; font-weight: bold;")
        self.diorama_result_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diorama_result_hint_label.hide()
        
        self.diorama_score_display_widget = CircularScore()
        self.diorama_score_display_widget.setFixedSize(180, 180)
        
        retry_diorama_button = QPushButton("ҚАЙТАЛАУ")
        retry_diorama_button.setFixedSize(180, 60)
        retry_diorama_button.setStyleSheet("QPushButton { background: rgba(255,255,255,0.1); color: white; border-radius: 20px; font-weight: bold; font-size: 18px; border: 1px solid rgba(255,255,255,0.2);}")
        retry_diorama_button.clicked.connect(lambda: self.diorama_overlay_view_stack.setCurrentIndex(0))
        
        self.save_diorama_result_button = QPushButton("НӘТИЖЕНІ САҚТАУ ✓")
        self.save_diorama_result_button.setFixedSize(220, 60)
        self.save_diorama_result_button.setStyleSheet("QPushButton { background: #00F260; color: black; border-radius: 20px; font-weight: bold; font-size: 18px; border:none;} QPushButton:disabled { background: #555; color: #888; }")
        self.save_diorama_result_button.clicked.connect(self.on_diorama_save_result_clicked)
        
        diorama_buttons_layout = QHBoxLayout()
        diorama_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        diorama_buttons_layout.setSpacing(20)
        diorama_buttons_layout.addWidget(retry_diorama_button)
        diorama_buttons_layout.addWidget(self.save_diorama_result_button)
        
        results_page_layout.addWidget(self.diorama_result_subtitle_label)
        results_page_layout.addLayout(self.diorama_letters_breakdown_layout)
        results_page_layout.addSpacing(10)
        results_page_layout.addWidget(self.diorama_result_hint_label, alignment=Qt.AlignmentFlag.AlignCenter)
        results_page_layout.addStretch()
        results_page_layout.addWidget(self.diorama_score_display_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        results_page_layout.addSpacing(30)
        results_page_layout.addLayout(diorama_buttons_layout)
        
        self.diorama_overlay_view_stack.addWidget(start_recording_page)
        self.diorama_overlay_view_stack.addWidget(active_recording_page)
        self.diorama_overlay_view_stack.addWidget(results_page)
        overlay_inner_layout.addWidget(self.diorama_overlay_view_stack)

        self.diorama_feedback_overlay = FeedbackOverlay(self.diorama_view_page)

        self.live_description_page = QWidget()
        live_page_layout = QVBoxLayout(self.live_description_page)
        live_page_layout.setContentsMargins(0, 0, 0, 0)
        
        self.live_background_video_label = QLabel(self.live_description_page)
        self.live_background_video_label.setScaledContents(True)
        self.live_background_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_background_video_label.setStyleSheet("background: #000;")
        
        self.background_media_player = QMediaPlayer(self)
        self.background_audio_output = QAudioOutput(self)
        self.background_audio_output.setVolume(0.05) 
        self.background_media_player.setAudioOutput(self.background_audio_output)
        
        self.live_ui_overlay_container = QWidget(self.live_description_page)
        self.live_ui_overlay_container.setStyleSheet("background: transparent;") 
        
        live_ui_layout = QVBoxLayout(self.live_ui_overlay_container)
        live_ui_layout.setContentsMargins(30, 30, 30, 30)
        
        live_top_bar_layout = QHBoxLayout()
        back_from_live_button = QPushButton("⬅ МӘЗІР")
        back_from_live_button.setFixedSize(120, 45)
        back_from_live_button.setStyleSheet("QPushButton { background: rgba(0, 0, 0, 150); color: white; border-radius: 20px; font-weight: bold; font-size: 14px; border: 1px solid rgba(255,255,255,40);} QPushButton:hover { background: rgba(255,255,255,0.2); }")
        back_from_live_button.clicked.connect(self.close_live_description_level)
        
        self.live_level_title_label = QLabel("...")
        self.live_level_title_label.setStyleSheet("font-size: 28px; font-weight: 900; color: white; background: rgba(0,0,0,150); padding: 5px 20px; border-radius: 15px;")
        
        live_top_bar_layout.addWidget(back_from_live_button)
        live_top_bar_layout.addStretch()
        live_top_bar_layout.addWidget(self.live_level_title_label)
        live_top_bar_layout.addStretch()
        live_top_bar_layout.addSpacing(120) 
        
        live_ui_layout.addLayout(live_top_bar_layout)
        live_ui_layout.addStretch()
        
        live_middle_area_layout = QHBoxLayout()
        
        self.live_missions_panel_frame = QFrame()
        self.live_missions_panel_frame.setFixedWidth(280)
        self.live_missions_panel_frame.setStyleSheet("background: rgba(0, 0, 0, 150); border-radius: 20px;")
        self.live_missions_panel_layout = QVBoxLayout(self.live_missions_panel_frame)
        self.live_missions_panel_layout.setContentsMargins(20, 20, 20, 20)
        
        self.live_missions_instruction_label = QLabel("Осы кілт сөздерді қолданып\nайналаны сипаттаңыз:")
        self.live_missions_instruction_label.setWordWrap(True)
        self.live_missions_instruction_label.setStyleSheet("color: #F59E0B; font-weight: 900; font-size: 14px; background: transparent; border: none;")
        self.live_missions_panel_layout.addWidget(self.live_missions_instruction_label)
        self.mission_keyword_ui_labels = {}
        
        live_middle_area_layout.addWidget(self.live_missions_panel_frame, alignment=Qt.AlignmentFlag.AlignTop)
        live_middle_area_layout.addStretch()
        
        self.live_results_scroll_area = QScrollArea()
        self.live_results_scroll_area.setFixedWidth(420)
        self.live_results_scroll_area.setWidgetResizable(True)
        self.live_results_scroll_area.setStyleSheet("background: rgba(20,20,25,100); border-radius: 20px; border: 1px solid rgba(255,255,255,20);")
        
        self.live_results_container_widget = QWidget()
        self.live_results_container_widget.setStyleSheet("background: transparent;")
        self.live_results_layout = QVBoxLayout(self.live_results_container_widget)
        self.live_results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.live_results_layout.setSpacing(15)
        
        self.live_results_scroll_area.setWidget(self.live_results_container_widget)
        live_middle_area_layout.addWidget(self.live_results_scroll_area, alignment=Qt.AlignmentFlag.AlignTop)
        
        live_ui_layout.addLayout(live_middle_area_layout)
        live_ui_layout.addStretch()
        
        live_bottom_area_layout = QHBoxLayout()
        self.toggle_live_recording_button = QPushButton("🎙 СИПАТТАУДЫ БАСТАУ")
        self.toggle_live_recording_button.setFixedSize(350, 60)
        self.toggle_live_recording_button.setStyleSheet("QPushButton { background: #F59E0B; border-radius: 30px; color: white; font-size: 20px; font-weight: bold;} QPushButton:checked { background: #EF4444; }")
        self.toggle_live_recording_button.setCheckable(True)
        self.toggle_live_recording_button.clicked.connect(self.toggle_live_mode_recording)
        live_bottom_area_layout.addStretch()
        live_bottom_area_layout.addWidget(self.toggle_live_recording_button)
        live_bottom_area_layout.addStretch()
        
        live_ui_layout.addLayout(live_bottom_area_layout)

        self.live_feedback_overlay = FeedbackOverlay(self.live_description_page)
        
        self.live_word_details_overlay = QFrame(self.live_description_page)
        self.live_word_details_overlay.setStyleSheet("QFrame { background: rgba(20, 20, 25, 240); border-radius: 40px; border: 2px solid #F59E0B; }")
        self.live_word_details_overlay.hide()
        details_overlay_layout = QVBoxLayout(self.live_word_details_overlay)
        details_overlay_layout.setContentsMargins(40, 40, 40, 40)
        
        details_top_bar_layout = QHBoxLayout()
        self.live_overlay_target_word_label = QLabel("")
        self.live_overlay_target_word_label.setStyleSheet("font-size: 42px; font-weight: 900; color: white; background: transparent; border: none;")
        close_details_button = QPushButton("✖")
        close_details_button.setFixedSize(50,50)
        close_details_button.setStyleSheet("QPushButton { background: #EF4444; border-radius: 25px; color: white; font-weight: bold; font-size: 20px; border: none;}")
        close_details_button.clicked.connect(self.close_live_word_details_overlay)
        details_top_bar_layout.addWidget(self.live_overlay_target_word_label)
        details_top_bar_layout.addStretch()
        details_top_bar_layout.addWidget(close_details_button)
        details_overlay_layout.addLayout(details_top_bar_layout)
        
        self.live_overlay_letters_breakdown_layout = QHBoxLayout()
        self.live_overlay_letters_breakdown_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_overlay_letters_breakdown_layout.setSpacing(10)
        details_overlay_layout.addLayout(self.live_overlay_letters_breakdown_layout)
        
        live_overlay_hint_label = QLabel("ℹ️ Қатені көру үшін әріпті басыңыз")
        live_overlay_hint_label.setStyleSheet("color: rgba(255,255,255,180); font-size: 14px; background: rgba(0,0,0,100); border-radius: 10px; padding: 5px 15px; font-weight: bold;")
        live_overlay_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        details_overlay_layout.addSpacing(10)
        details_overlay_layout.addWidget(live_overlay_hint_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.live_overlay_circular_score_widget = CircularScore()
        self.live_overlay_circular_score_widget.setFixedSize(180, 180)
        details_overlay_layout.addStretch()
        details_overlay_layout.addWidget(self.live_overlay_circular_score_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.save_live_word_result_button = QPushButton("НӘТИЖЕНІ САҚТАУ ✓")
        self.save_live_word_result_button.setFixedSize(220, 60)
        self.save_live_word_result_button.setStyleSheet("QPushButton { background: #00F260; color: black; border-radius: 20px; font-weight: bold; font-size: 18px; border:none;} QPushButton:disabled { background: #555; color: #888; }")
        self.save_live_word_result_button.clicked.connect(self.on_live_word_save_clicked)
        details_overlay_layout.addSpacing(30)
        details_overlay_layout.addWidget(self.save_live_word_result_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.statistics_view_page = ProfileStatsPage(lambda: self.view_stack.setCurrentIndex(0))

        self.view_stack.addWidget(main_menu_widget)                  
        self.view_stack.addWidget(self.training_page_widget)       
        self.view_stack.addWidget(self.interactive_level_selector_page) 
        self.view_stack.addWidget(self.diorama_view_page)     
        self.view_stack.addWidget(self.live_level_selector_page)  
        self.view_stack.addWidget(self.live_description_page)        
        self.view_stack.addWidget(self.statistics_view_page)       
        
        self.main_layout.addWidget(self.view_stack)

    def exit_training_page(self):
        if getattr(self, 'is_recording_audio', False):
            self.toggle_audio_recording()
        self.training_feedback_overlay.hide_overlay()
        self.circular_score_widget.set_score(0)
        self.main_save_result_button.hide()
        self.view_stack.setCurrentIndex(0)

    def set_application_mode(self, new_mode):
        self.current_app_mode = new_mode
        self.view_stack.setCurrentIndex(1)
        self.circular_score_widget.set_score(0)
        self.main_save_result_button.hide()
        self.word_mode_hint_label.hide()
        self.score_hint_label.setText("ℹ Қатені көру")
        
        while self.word_breakdown_layout.count():
            widget_item = self.word_breakdown_layout.takeAt(0)
            if widget_item.widget():
                widget_item.widget().deleteLater()
                
        if new_mode == "word": 
            self.selection_title_label.setText("ТАҢДАЛҒАН СӨЗ:")
            self.letter_combo_box.hide()
            self.word_combo_box.show()
            self.word_breakdown_container_widget.show()
            self.score_hint_label.hide()
            self.on_word_selection_changed()
        else: 
            self.selection_title_label.setText("ТАҢДАЛҒАН ӘРІП:")
            self.letter_combo_box.show()
            self.word_combo_box.hide()
            self.word_breakdown_container_widget.hide()
            self.score_hint_label.show()
            self.on_letter_selection_changed()
            
        self.reposition_floating_ui_elements()

    def on_letter_selection_changed(self): 
        pass

    def on_word_selection_changed(self): 
        pass

    def on_score_widget_clicked(self, event):
        if self.current_app_mode == "letter" and hasattr(self, 'last_letter_char'):
            diagnostic_advice = get_diagnostic_advice(self.last_letter_char, self.last_letter_err, self.last_letter_score)
            self.training_feedback_overlay.show_feedback(self.last_letter_char, self.last_letter_score, diagnostic_advice)

    def check_and_display_save_button(self):
        if self.settings_overlay_widget.app_settings.get("auto_save", False):
            target_text_to_save = self.current_target_word_to_save if getattr(self, 'current_app_mode', '') == "word" else self.letter_combo_box.currentText()
            save_to_db(target_text_to_save, self.last_calculated_final_score)
            self.statistics_view_page.load_data()
            self.main_save_result_button.hide()
        else:
            self.main_save_result_button.setText("НӘТИЖЕНІ САҚТАУ ✓")
            self.main_save_result_button.setEnabled(True)
            self.main_save_result_button.show()

    def handle_save_result_click(self):
        target_text_to_save = self.current_target_word_to_save if getattr(self, 'current_app_mode', '') == "word" else self.letter_combo_box.currentText()
        save_to_db(target_text_to_save, self.last_calculated_final_score)
        self.statistics_view_page.load_data()
        self.main_save_result_button.setText("САҚТАЛДЫ ✓")
        self.main_save_result_button.setEnabled(False)

    def on_diorama_save_result_clicked(self):
        save_to_db(self.current_target_word_to_save, self.last_calculated_final_score)
        self.statistics_view_page.load_data()
        self.save_diorama_result_button.setText("САҚТАЛДЫ ✓")
        self.save_diorama_result_button.setEnabled(False)

    def exit_diorama_mode(self):
        if getattr(self, 'is_diorama_mode_recording', False):
            self.stop_diorama_audio_recording()
        self.diorama_feedback_overlay.hide_overlay()
        self.diorama_overlay_view_stack.setCurrentIndex(0)
        self.diorama_action_overlay.hide()
        self.view_stack.setCurrentIndex(2)

    def open_diorama_overlay(self, selected_word):
        self.current_app_mode = "diorama"
        self.current_target_word_to_save = selected_word
        self.diorama_target_word_label.setText(selected_word)
        self.diorama_word_blur_effect.setEnabled(True)
        self.diorama_hint_button.show()
        self.diorama_result_hint_label.hide()
        self.diorama_overlay_view_stack.setCurrentIndex(0)
        overlay_width, overlay_height = 800, 600
        self.diorama_action_overlay.setGeometry((self.width() - overlay_width) // 2, (self.height() - overlay_height) // 2, overlay_width, overlay_height)
        self.diorama_action_overlay.show()

    def close_diorama_overlay_window(self): 
        if getattr(self, 'is_diorama_mode_recording', False):
            self.stop_diorama_audio_recording()
        self.diorama_feedback_overlay.hide_overlay()
        self.diorama_overlay_view_stack.setCurrentIndex(0)
        self.diorama_action_overlay.hide()

    def toggle_camera_visibility(self): 
        self.is_camera_hidden = not self.is_camera_hidden
        if hasattr(self, 'oval_face_overlay'):
            if self.is_camera_hidden:
                self.oval_face_overlay.hide()
            else:
                self.oval_face_overlay.show()

    def process_camera_frame(self, camera_frame, face_landmarks, calculated_lip_ratio, is_face_aligned):
        frame_height, frame_width, _ = camera_frame.shape
        if self.is_camera_hidden: 
            camera_frame[:] = 15
            self.oval_face_overlay.set_aligned(False)
        else:
            if face_landmarks:
                if self.is_recording_audio:
                    self.lip_activity_history.append(calculated_lip_ratio)
                self.draw_ar_visual_markers(camera_frame, face_landmarks, frame_width, frame_height, calculated_lip_ratio)
            if hasattr(self, 'oval_face_overlay'):
                self.oval_face_overlay.set_aligned(is_face_aligned)
        
        qt_image_format = QImage(camera_frame.data, frame_width, frame_height, frame_width * 3, QImage.Format.Format_BGR888).copy()
        scaled_pixmap = QPixmap.fromImage(qt_image_format).scaled(self.video_feed_label.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.FastTransformation)
        self.video_feed_label.setPixmap(scaled_pixmap)

    def draw_ar_visual_markers(self, camera_frame, landmark_points, frame_width, frame_height, current_lip_ratio):
        if getattr(self, 'current_app_mode', '') in ["word", "diorama"]:
            marker_color = (255, 200, 0)
        else:
            target_letter = self.letter_combo_box.currentText()
            target_mouth_distance = TARGET_MOUTH_OPENNESS.get(target_letter, 0.03)
            lip_accuracy_percentage = max(0, 100 - abs(current_lip_ratio - target_mouth_distance) * 1000)
            marker_color = (0, 242, 96) if lip_accuracy_percentage > 85 else (43, 75, 255)
            
        lip_landmark_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 13, 14]
        for index in lip_landmark_indices: 
            cv2.circle(camera_frame, (int(landmark_points[index].x * frame_width), int(landmark_points[index].y * frame_height)), 1, marker_color, -1)

    def audio_stream_callback(self, input_data, frame_count, time_info, status_flags):
        volume_norm = np.linalg.norm(input_data) * 10
        self.current_volume_level = min(1.0, volume_norm)
        if self.is_recording_audio or getattr(self, 'is_diorama_mode_recording', False):
            self.audio_capture_data.extend(input_data[:,0])
        elif getattr(self, 'is_live_mode_recording', False):
            self.live_transcription_thread.add_data(input_data[:,0].copy())

    def toggle_audio_recording(self):
        if self.record_audio_button.is_currently_processing:
            return
            
        if self.record_audio_button.isChecked():
            self.is_recording_audio = True
            self.main_save_result_button.hide()
            self.word_mode_hint_label.hide()
            self.audio_capture_data = []
            self.lip_activity_history = []
            self.circular_score_widget.set_score(0)
            
            while self.word_breakdown_layout.count():
                widget_item = self.word_breakdown_layout.takeAt(0)
                if widget_item.widget():
                    widget_item.widget().deleteLater()
                    
            self.active_audio_stream = sd.InputStream(samplerate=16000, channels=1, callback=self.audio_stream_callback)
            self.active_audio_stream.start()
        else:
            self.is_recording_audio = False
            self.record_audio_button.set_processing(True)
            self.score_hint_label.setText("⏳ Күтіңіз...")
            QTimer.singleShot(300, self.process_stopped_recording)

    def process_stopped_recording(self):
        if hasattr(self, 'active_audio_stream') and self.active_audio_stream is not None:
            self.active_audio_stream.stop()
            self.active_audio_stream.close()
            
        if getattr(self, 'current_app_mode', '') == "word":
            self.calculate_full_word_score()
        else:
            self.calculate_single_letter_score()

    def calculate_single_letter_score(self):
        audio_array = np.array(self.audio_capture_data, dtype=np.float32)
        target_letter = self.letter_combo_box.currentText()
        
        max_volume = np.max(np.abs(audio_array)) if audio_array.size > 0 else 0
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_array)) if audio_array.size > 0 else 0
        
        if audio_array.size == 0 or max_volume < 0.004: 
            self.record_audio_button.set_processing(False)
            self.record_audio_button.setChecked(False)
            self.animate_score_display(0)
            self.last_letter_char = target_letter
            self.last_letter_score = 0
            self.last_letter_err = "volume"
            self.score_hint_label.setText("ℹ Қатені көру")
            self.training_feedback_overlay.show_feedback("!", 0, "Дыбыс өте ақырын. Қаттырақ сөйлеңіз.")
            return 
            
        audio_score, _ = self.audio_engine.calculate_phoneme_score(audio_array, target_letter)
        
        if target_letter in ["Ш", "С", "Щ", "Ф", "Х", "Һ"]:
            if zero_crossing_rate > 0.06: 
                audio_score = min(100.0, audio_score + 30.0) 
            elif zero_crossing_rate < 0.02:
                audio_score = min(audio_score, 30.0)
                
        elif target_letter == "Қ":
            if max_volume > 0.045:
                audio_score = min(100.0, audio_score + 25.0)
            elif max_volume < 0.01:
                audio_score = min(audio_score, 40.0)
                
        elif target_letter == "Ғ":
            if zero_crossing_rate > 0.09:
                audio_score = min(audio_score, 40.0)
            elif max_volume > 0.015 and zero_crossing_rate < 0.07:
                audio_score = min(100.0, audio_score + 20.0)
                
        elif target_letter in ["Ң", "М", "Н"]:
            if zero_crossing_rate > 0.08:
                audio_score = min(audio_score, 35.0)
            elif max_volume > 0.015:
                audio_score = min(100.0, audio_score + 15.0)
                
        elif target_letter in ["П", "Б", "Т", "Д", "К", "Г"]:
            if max_volume > 0.035:
                audio_score = min(100.0, audio_score + 15.0)
                
        elif target_letter in ["А", "О", "У", "Э", "И", "Ы", "Е", "Ә", "Ө", "Ұ", "Ү"]:
            if zero_crossing_rate > 0.15:
                audio_score = min(audio_score, 40.0)
            elif max_volume > 0.03:
                audio_score = min(100.0, audio_score + 10.0)

        target_lip_ratio = TARGET_MOUTH_OPENNESS.get(target_letter, 0.03)
        if target_letter in ["П", "Б", "М"]: 
            current_lip_ratio = np.min(self.lip_activity_history) if self.lip_activity_history else 0.0
        else: 
            current_lip_ratio = np.max(self.lip_activity_history) if self.lip_activity_history else 0.0
            
        lip_score = max(0, 1 - abs(current_lip_ratio - target_lip_ratio) / 0.40) * 100
        
        error_code = "general"
        if lip_score < 40: 
            error_code = "lips"
        
        if audio_score < 15.0:  
            final_score = audio_score 
            error_code = "acoustic"
        else: 
            audio_weight_factor = 0.85 if target_letter not in ["А","О","У","Э","И","Ы","Е","Ә","Ө","Ұ","Ү"] else 0.7
            final_score = (audio_score * audio_weight_factor) + (lip_score * (1.0 - audio_weight_factor))
            if final_score >= 80: 
                error_code = "perfect"

        if target_letter in CONFUSION_GROUPS and audio_array.size > 1000 and np.max(np.abs(audio_array)) >= 0.01:
            best_confusion_score = 0
            best_confusion_char = None
            for confusion_char in CONFUSION_GROUPS[target_letter]:
                confusion_score, _ = self.audio_engine.calculate_phoneme_score(audio_array, confusion_char)
                if confusion_score > best_confusion_score: 
                    best_confusion_score = confusion_score
                    best_confusion_char = confusion_char
                    
            if best_confusion_score > 40 and best_confusion_score > audio_score:
                final_score = max(0, min(45, final_score * 0.5))
                error_code = f"sub_{best_confusion_char}"

        final_score = self.apply_soft_boost(final_score)
        if final_score > 98: 
            final_score = 98
        
        self.record_audio_button.set_processing(False)
        self.record_audio_button.setChecked(False)
        self.animate_score_display(final_score)
        self.score_hint_label.setText("ℹ Қатені көру")
        self.last_letter_char = target_letter
        self.last_letter_score = int(final_score)
        self.last_letter_err = error_code

    def calculate_full_word_score(self):
        audio_array = np.array(self.audio_capture_data, dtype=np.float32)
        target_word = self.word_combo_box.currentText().strip().upper()
        self.current_target_word_to_save = target_word
        
        rms_volume = np.sqrt(np.mean(audio_array**2)) if len(audio_array) > 0 else 0.0
        if len(audio_array) < 4000 or rms_volume < 0.003:
            self.process_word_analysis_finish("", 0.0, target_word, "word", "Дыбыс естілмеді. Қаттырақ сөйлеңіз.")
            return
            
        if self.audio_engine.vosk_model is None:
            self.process_word_analysis_finish("", 0.0, target_word, "word", "Vosk Error")
            return
            
        audio_array = preprocess_audio_signal(audio_array)
        self.word_transcription_thread = TranscribeThread(self.audio_engine.vosk_model, audio_array, target_word, "word")
        copied_lip_activity = list(self.lip_activity_history)
        self.word_transcription_thread.finished.connect(lambda text, conf, target, mode: self.process_word_analysis_finish(text, conf, target, mode, raw_audio=audio_array.copy(), lip_activity_array=copied_lip_activity))
        self.word_transcription_thread.start()

    def process_word_analysis_finish(self, recognized_text, confidence, target_word, origin_mode, custom_error_message="", raw_audio=None, lip_activity_array=None):
        self.record_audio_button.set_processing(False)
        self.record_audio_button.setChecked(False)
        self.record_audio_button.setEnabled(True)
        self.score_hint_label.setText("ℹ Қатені көру")
        
        if custom_error_message:
            self.training_feedback_overlay.show_feedback("!", 0, custom_error_message)
            return
            
        target_no_spaces = target_word.replace(" ", "")
        recognized_no_spaces = recognized_text.replace(" ", "")
        
        safe_audio_array = raw_audio if raw_audio is not None and len(raw_audio) > 0 else np.array([])
        letter_results = self.align_and_score_phonemes(target_no_spaces, recognized_no_spaces, confidence, safe_audio_array, lip_activity_array)
        
        self.word_breakdown_layout.addStretch()
        self.pending_animation_widgets = []
        score_index = 0
        
        for character in target_word:
            if character == " ":
                space_label = QLabel()
                space_label.setFixedWidth(20)
                space_label.setStyleSheet("background: transparent; border: none;")
                space_label.hide()
                self.word_breakdown_layout.addWidget(space_label)
                self.pending_animation_widgets.append((character, space_label))
            else:
                score, error = letter_results[score_index]
                letter_card = InteractiveLetterCard(character, score, error)
                letter_card.card_clicked.connect(self.training_feedback_overlay.show_feedback)
                letter_card.hide()
                self.word_breakdown_layout.addWidget(letter_card)
                self.pending_animation_widgets.append((character, letter_card))
                score_index += 1
                
        self.word_breakdown_layout.addStretch()
        self.pending_final_word_score = sum(result[0] for result in letter_results) // len(letter_results) if letter_results else 0
        self.letter_animation_index = 0
        self.display_letters_sequentially()

    def align_and_score_phonemes(self, target_word, recognized_word, confidence, raw_audio, lip_activity_array):
        letter_results = [] 
        if not recognized_word: 
            return [(0, "volume")] * len(target_word)
            
        sequence_matcher = difflib.SequenceMatcher(None, target_word, recognized_word)
        target_to_recognized_map = {}
        for tag, i1, i2, j1, j2 in sequence_matcher.get_opcodes():
            if tag in ['equal', 'replace']:
                for target_index, recognized_index in zip(range(i1, i2), range(j1, j2)): 
                    target_to_recognized_map[target_index] = recognized_index
            elif tag == 'delete':
                for target_index in range(i1, i2): 
                    target_to_recognized_map[target_index] = -1 
                    
        total_spoken_letters = max(1, len(recognized_word))
        base_confidence_score = max(40, min(98, confidence * 100))
        matched_phoneme_metrics = []
        
        for i, char in enumerate(target_word):
            recognized_index = target_to_recognized_map.get(i, -1)
            if recognized_index != -1 and recognized_index < len(recognized_word) and recognized_word[recognized_index] == char:
                start_ratio = recognized_index / total_spoken_letters
                end_ratio = (recognized_index + 1) / total_spoken_letters
                audio_slice = raw_audio[int(start_ratio * len(raw_audio)) : int(end_ratio * len(raw_audio))]
                rms_volume = np.sqrt(np.mean(audio_slice**2)) if len(audio_slice) > 0 else 0.0
                current_lip_ratio = 0.03
                
                if lip_activity_array:
                    start_lip_index = int(start_ratio * len(lip_activity_array))
                    end_lip_index = start_lip_index + max(1, int(len(lip_activity_array)/total_spoken_letters))
                    current_lip_ratio = np.mean(lip_activity_array[start_lip_index:end_lip_index])
                    
                target_lip_ratio = TARGET_MOUTH_OPENNESS.get(char, 0.03)
                lip_accuracy = max(0, 1 - abs(current_lip_ratio - target_lip_ratio) / 0.10) * 100
                matched_phoneme_metrics.append((rms_volume, lip_accuracy))
                
        average_rms = np.mean([metric[0] for metric in matched_phoneme_metrics]) if matched_phoneme_metrics else 0.0
        average_lip_score = np.mean([metric[1] for metric in matched_phoneme_metrics]) if matched_phoneme_metrics else base_confidence_score
        
        for i, char in enumerate(target_word):
            recognized_index = target_to_recognized_map.get(i, -1)
            if recognized_index == -1: 
                letter_results.append((0, "volume"))
                continue
                
            start_ratio = recognized_index / total_spoken_letters
            end_ratio = (recognized_index + 1) / total_spoken_letters
            audio_slice = raw_audio[int(start_ratio * len(raw_audio)) : int(end_ratio * len(raw_audio))]
            rms_volume = np.sqrt(np.mean(audio_slice**2)) if len(audio_slice) > 0 else 0.0
            current_lip_ratio = 0.03
            
            if lip_activity_array:
                start_lip_index = int(start_ratio * len(lip_activity_array))
                end_lip_index = start_lip_index + max(1, int(len(lip_activity_array)/total_spoken_letters))
                current_lip_ratio = np.mean(lip_activity_array[start_lip_index:end_lip_index])
                
            target_lip_ratio = TARGET_MOUTH_OPENNESS.get(char, 0.03)
            lip_accuracy = max(0, 1 - abs(current_lip_ratio - target_lip_ratio) / 0.10) * 100
            recognized_char = recognized_word[recognized_index] if 0 <= recognized_index < len(recognized_word) else None
            
            if recognized_char == char:
                relative_score = base_confidence_score + ((lip_accuracy - average_lip_score) * 0.4) + (((rms_volume - average_rms) / (average_rms + 1e-6)) * 15)
                final_letter_score = max(50, min(98, relative_score))
                error_code = "perfect" if final_letter_score >= 80 else ("lips" if lip_accuracy < 50 else "acoustic")
                
                if char in CONFUSION_GROUPS and audio_slice.size > 1000 and np.max(np.abs(audio_slice)) >= 0.01:
                    best_confusion_score = 0
                    best_confusion_char = None
                    for confusion_char in CONFUSION_GROUPS[char]:
                        confusion_score, _ = self.audio_engine.calculate_phoneme_score(audio_slice, confusion_char)
                        if confusion_score > best_confusion_score: 
                            best_confusion_score = confusion_score
                            best_confusion_char = confusion_char
                            
                    if best_confusion_score > 30 and best_confusion_score > self.audio_engine.calculate_phoneme_score(audio_slice, char)[0]:
                        final_letter_score = max(30, min(50, relative_score * 0.5))
                        error_code = f"sub_{best_confusion_char}"
            else: 
                final_letter_score = max(10, min(55, base_confidence_score * 0.5))
                error_code = f"sub_{recognized_char}" if recognized_char else "general"
                
            final_letter_score = self.apply_soft_boost(final_letter_score)
            letter_results.append((int(final_letter_score), error_code))
            
        return letter_results

    def start_diorama_audio_recording(self):
        self.is_diorama_mode_recording = True
        self.diorama_overlay_view_stack.setCurrentIndex(1)
        self.diorama_score_display_widget.set_score(0)
        
        while self.diorama_letters_breakdown_layout.count():
            widget_item = self.diorama_letters_breakdown_layout.takeAt(0)
            if widget_item.widget():
                widget_item.widget().deleteLater()
                
        self.audio_capture_data = []
        self.active_audio_stream = sd.InputStream(samplerate=16000, channels=1, callback=self.audio_stream_callback)
        self.active_audio_stream.start()
    
    def stop_diorama_audio_recording(self): 
        self.is_diorama_mode_recording = False
        self.stop_diorama_recording_button.setText("⏳ КҮТІҢІЗ...")
        QTimer.singleShot(300, self.process_stopped_diorama_recording)
        
    def process_stopped_diorama_recording(self): 
        self.stop_diorama_recording_button.setText("ТОҚТАТУ")
        if hasattr(self, 'active_audio_stream') and self.active_audio_stream is not None:
            self.active_audio_stream.stop()
        self.calculate_diorama_word_score()
        
    def calculate_diorama_word_score(self):
        audio_array = np.array(self.audio_capture_data, dtype=np.float32)
        target_word = self.current_target_word_to_save
        rms_volume = np.sqrt(np.mean(audio_array**2)) if len(audio_array) > 0 else 0.0
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_array)) if len(audio_array) > 0 else 0.0
        
        if len(audio_array) < 4000 or (rms_volume < 0.003 and zero_crossing_rate < 0.04): 
            self.finish_diorama_word_analysis("", 0.0, target_word, custom_error="Дыбыс естілмеді. Қаттырақ сөйлеңіз.", raw_audio=audio_array)
            return
            
        audio_array = preprocess_audio_signal(audio_array)
        self.diorama_transcription_thread = TranscribeThread(self.audio_engine.vosk_model, audio_array, target_word, "diorama")
        self.diorama_transcription_thread.finished.connect(lambda text, conf, target, mode: self.finish_diorama_word_analysis(text, conf, target, raw_audio=audio_array.copy()))
        self.diorama_transcription_thread.start()
        
    def finish_diorama_word_analysis(self, recognized_text, confidence, target_word, custom_error="", raw_audio=None):
        if custom_error:
            self.diorama_feedback_overlay.show_feedback("!", 0, custom_error)
            self.diorama_overlay_view_stack.setCurrentIndex(0)
            return
            
        safe_audio_array = raw_audio if raw_audio is not None and len(raw_audio) > 0 else np.array([])
        letter_results = self.align_and_score_phonemes(target_word.replace(" ",""), recognized_text.replace(" ",""), confidence, safe_audio_array, [])
        score_index = 0
        calculated_letter_scores = []
        
        for character in target_word:
            if character == " ": 
                space_label = QLabel(" ")
                space_label.setFixedWidth(20)
                self.diorama_letters_breakdown_layout.addWidget(space_label)
            else:
                score, error = letter_results[score_index]
                calculated_letter_scores.append(score)
                letter_card = InteractiveLetterCard(character, score, error)
                letter_card.card_clicked.connect(self.diorama_feedback_overlay.show_feedback)
                self.diorama_letters_breakdown_layout.addWidget(letter_card)
                score_index += 1
                
        self.reveal_diorama_hidden_word()
        self.diorama_overlay_view_stack.setCurrentIndex(2)
        self.diorama_result_hint_label.show()
        
        final_word_score = sum(calculated_letter_scores) // len(calculated_letter_scores) if recognized_text else 0
        self.last_calculated_final_score = final_word_score
        
        if self.settings_overlay_widget.app_settings.get("auto_save", False):
            save_to_db(self.current_target_word_to_save, final_word_score)
            self.statistics_view_page.load_data()
            self.save_diorama_result_button.hide()
        else:
            self.save_diorama_result_button.show()
            self.save_diorama_result_button.setEnabled(True)
            self.save_diorama_result_button.setText("НӘТИЖЕНІ САҚТАУ ✓")
            
        self.animate_score_display(final_word_score)
    
    def reveal_diorama_hidden_word(self):
        self.diorama_word_blur_effect.setEnabled(False)
        self.diorama_hint_button.hide()
        
    def open_diorama_level(self, level_data):
        self.diorama_interactive_scene.load_location(level_data["folder"])
        self.view_stack.setCurrentIndex(3)

    def on_live_word_detected(self, target_word, confidence, audio_slice):
        safe_audio_array = preprocess_audio_signal(audio_slice)
        letter_results = self.align_and_score_phonemes(target_word.replace(" ",""), target_word.replace(" ",""), confidence, safe_audio_array, [])
        final_word_score = sum(result[0] for result in letter_results) // len(letter_results) if letter_results else 0
        
        if target_word in self.mission_keyword_ui_labels:
            self.mission_keyword_ui_labels[target_word].setText("✅ " + target_word)
            self.mission_keyword_ui_labels[target_word].setStyleSheet("color: #00F260; font-weight: bold;")
            
        word_card = LiveWordCard(target_word, final_word_score, letter_results, self.show_live_word_details)
        self.live_results_layout.addWidget(word_card)
        self.smooth_scroll_live_results()

    def show_live_word_details(self, word, score, letter_results):
        self.current_target_word_to_save = word
        self.last_calculated_final_score = score
        self.live_overlay_target_word_label.setText(word)
        
        while self.live_overlay_letters_breakdown_layout.count():
            widget_item = self.live_overlay_letters_breakdown_layout.takeAt(0)
            if widget_item.widget():
                widget_item.widget().deleteLater()
            
        score_index = 0
        for character in word:
            if character == " ":
                space_label = QLabel(" ")
                space_label.setFixedWidth(20)
                self.live_overlay_letters_breakdown_layout.addWidget(space_label)
            else:
                letter_score, error_code = letter_results[score_index]
                letter_card = InteractiveLetterCard(character, letter_score, error_code)
                letter_card.card_clicked.connect(self.live_feedback_overlay.show_feedback)
                self.live_overlay_letters_breakdown_layout.addWidget(letter_card)
                score_index += 1
                
        self.live_overlay_circular_score_widget.set_score(score)
        
        if self.settings_overlay_widget.app_settings.get("auto_save", False):
            self.save_live_word_result_button.hide()
        else:
            self.save_live_word_result_button.setText("НӘТИЖЕНІ САҚТАУ ✓")
            self.save_live_word_result_button.setEnabled(True)
            self.save_live_word_result_button.show()
            
        overlay_width, overlay_height = 800, 600
        self.live_word_details_overlay.setGeometry((self.width() - overlay_width) // 2, (self.height() - overlay_height) // 2, overlay_width, overlay_height)
        self.live_word_details_overlay.raise_()
        self.live_word_details_overlay.show()

    def close_live_word_details_overlay(self):
        self.live_word_details_overlay.hide()
        self.live_feedback_overlay.hide_overlay()

    def on_live_word_save_clicked(self):
        save_to_db(self.current_target_word_to_save, self.last_calculated_final_score)
        self.statistics_view_page.load_data()
        self.save_live_word_result_button.setText("САҚТАЛДЫ ✓")
        self.save_live_word_result_button.setEnabled(False)

    def smooth_scroll_live_results(self):
        QApplication.processEvents()
        if hasattr(self, 'live_results_scroll_area'):
            vertical_scrollbar = self.live_results_scroll_area.verticalScrollBar()
            self.smooth_scroll_animation = QVariantAnimation(self)
            self.smooth_scroll_animation.setDuration(800)
            self.smooth_scroll_animation.setStartValue(vertical_scrollbar.value())
            self.smooth_scroll_animation.setEndValue(vertical_scrollbar.maximum())
            self.smooth_scroll_animation.valueChanged.connect(vertical_scrollbar.setValue)
            self.smooth_scroll_animation.start()

    def toggle_live_mode_recording(self):
        if self.toggle_live_recording_button.isChecked():
            self.is_live_mode_recording = True
            self.toggle_live_recording_button.setText("ТОҚТАТУ ЖӘНЕ ТАЛДАУ")
            self.live_transcription_thread = LiveTranscribeThread(self.audio_engine.vosk_model, self.current_live_level_data["missions"])
            self.live_transcription_thread.word_found_signal.connect(self.on_live_word_detected)
            self.live_transcription_thread.start()
            self.audio_capture_data = []
            self.active_audio_stream = sd.InputStream(samplerate=16000, channels=1, blocksize=4000, callback=self.audio_stream_callback)
            self.active_audio_stream.start()
        else: 
            self.is_live_mode_recording = False
            self.toggle_live_recording_button.setText("🎙 СИПАТТАУДЫ БАСТАУ")
            if hasattr(self, 'active_audio_stream') and self.active_audio_stream is not None:
                self.active_audio_stream.stop()
            if hasattr(self, 'live_transcription_thread') and self.live_transcription_thread is not None:
                self.live_transcription_thread.stop()
            
    def open_live_level(self, level_data): 
        self.current_live_level_data = level_data
        self.live_level_title_label.setText(level_data["title"])
        self.background_media_player.stop()
        
        if hasattr(self, 'background_video_thread') and self.background_video_thread is not None:
            self.background_video_thread.stop()
            self.background_video_thread.deleteLater()
            self.background_video_thread = None
            
        while self.live_missions_panel_layout.count() > 1:
            widget_item = self.live_missions_panel_layout.takeAt(1)
            if widget_item.widget():
                widget_item.widget().deleteLater()
            
        self.mission_keyword_ui_labels.clear()
        for mission_word in level_data["missions"]:
            mission_label = QLabel("☐ " + mission_word)
            mission_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px; background: transparent; border: none; padding: 2px;")
            self.mission_keyword_ui_labels[mission_word] = mission_label
            self.live_missions_panel_layout.addWidget(mission_label)
            
        self.live_missions_panel_layout.addStretch()
        
        while self.live_results_layout.count():
            widget_item = self.live_results_layout.takeAt(0)
            if widget_item.widget():
                widget_item.widget().deleteLater()
        
        folder_name = os.path.basename(level_data["folder"])
        video_file_path = os.path.join(level_data["folder"], f"{folder_name}.mp4")
        if not os.path.exists(video_file_path): 
            video_file_path = os.path.join(level_data["folder"], "bg.mp4")
            
        if os.path.exists(video_file_path): 
            self.background_media_player.setSource(QUrl.fromLocalFile(video_file_path))
            self.background_media_player.play()
            self.background_video_thread = BgVideoThread(video_file_path)
            self.background_video_thread.change_pixmap_signal.connect(self.update_live_background_frame)
            self.background_video_thread.start()
            
        self.view_stack.setCurrentIndex(5)
        self.live_ui_overlay_container.raise_()
        
    def close_live_description_level(self): 
        self.background_media_player.stop()
        self.background_media_player.setSource(QUrl())
        if hasattr(self, 'background_video_thread') and self.background_video_thread is not None:
            self.background_video_thread.stop()
            self.background_video_thread.deleteLater()
            self.background_video_thread = None
            
        if getattr(self, 'is_live_mode_recording', False):
            self.toggle_live_mode_recording()
            
        self.live_feedback_overlay.hide_overlay()
        self.view_stack.setCurrentIndex(4)

    def update_live_background_frame(self, frame):
        self.current_background_frame = frame.copy() 
        height, width, channels = self.current_background_frame.shape
        qt_image_format = QImage(self.current_background_frame.data, width, height, channels * width, QImage.Format.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image_format).scaled(self.live_background_video_label.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.FastTransformation)
        self.live_background_video_label.setPixmap(scaled_pixmap)

    def show_statistics_page(self):
        self.statistics_view_page.load_data()
        self.view_stack.setCurrentIndex(6)
        
    def show_settings_overlay(self):
        self.settings_overlay_widget.raise_()
        self.settings_overlay_widget.show()

    def animate_score_display(self, final_score): 
        self.last_calculated_final_score = max(0, min(100, final_score))
        target_score_widget = self.diorama_score_display_widget if getattr(self,'current_app_mode','') == 'diorama' else self.circular_score_widget
        self.score_animation = QVariantAnimation(self, duration=1500, startValue=float(target_score_widget.current_display_value), endValue=float(final_score), easingCurve=QEasingCurve.Type.OutCubic, valueChanged=target_score_widget.set_score)
        self.score_animation.start()
        
        if getattr(self,'current_app_mode','') != 'diorama':
            self.score_animation.finished.connect(self.after_score_display_animation)
        
    def display_letters_sequentially(self):
        if self.letter_animation_index < len(self.pending_animation_widgets): 
            _, widget = self.pending_animation_widgets[self.letter_animation_index]
            widget.show()
            self.letter_animation_index += 1
            self.word_breakdown_container_widget.adjustSize()
            self.reposition_floating_ui_elements()
            QTimer.singleShot(120, self.display_letters_sequentially)
        else:
            self.animate_score_display(self.pending_final_word_score)
        
    def after_score_display_animation(self):
        if getattr(self, 'current_app_mode', '') == "word":
            self.word_mode_hint_label.show()
        if self.last_calculated_final_score >= 0: 
            self.check_and_display_save_button()
            
    def reposition_floating_ui_elements(self):
        widget_width, widget_height = self.width(), self.height()
        self.video_feed_label.setGeometry(0, 0, widget_width, widget_height)
        self.oval_face_overlay.setGeometry(0, 0, widget_width, widget_height)
        
        if hasattr(self, 'training_feedback_overlay'):
            self.training_feedback_overlay.setGeometry(0, 0, widget_width, widget_height)
        if hasattr(self, 'diorama_feedback_overlay'):
            self.diorama_feedback_overlay.setGeometry(0, 0, widget_width, widget_height)
        if hasattr(self, 'live_feedback_overlay'):
            self.live_feedback_overlay.setGeometry(0, 0, widget_width, widget_height)
        if hasattr(self, 'settings_overlay_widget'):
            self.settings_overlay_widget.setGeometry(0, 0, widget_width, widget_height)
            
        self.back_to_menu_button.move(30, 30)
        self.center_selection_pill.move(30, 90)
        self.camera_toggle_button.move(widget_width - 75, 30)
        self.score_display_pill.move(30, (widget_height - self.score_display_pill.height()) // 2)
        self.main_save_result_button.move(widget_width - 260, widget_height - 75)
        self.record_audio_button.move(30, widget_height - 130)
        self.voice_wave_display.move(150, widget_height - 100)
        self.word_breakdown_container_widget.move((widget_width - self.word_breakdown_container_widget.width()) // 2, widget_height - 140)
        
        if hasattr(self, 'word_mode_hint_label'):
            self.word_mode_hint_label.adjustSize()
            self.word_mode_hint_label.move((widget_width - self.word_mode_hint_label.width()) // 2, widget_height - 40)
            
        if hasattr(self, 'live_missions_panel_frame'):
            self.live_missions_panel_frame.move(30, 150)
            self.live_missions_panel_frame.adjustSize()
            
        if hasattr(self, 'live_word_details_overlay') and self.live_word_details_overlay.isVisible():
            overlay_width, overlay_height = 800, 600
            self.live_word_details_overlay.setGeometry((widget_width - overlay_width) // 2, (widget_height - overlay_height) // 2, overlay_width, overlay_height)
        
    def resizeEvent(self, event): 
        self.reposition_floating_ui_elements()
        if hasattr(self, 'live_background_video_label'): 
            self.live_background_video_label.setGeometry(0, 0, self.width(), self.height())
            self.live_ui_overlay_container.setGeometry(0, 0, self.width(), self.height())
            self.live_ui_overlay_container.raise_()
            if hasattr(self, 'live_word_details_overlay'):
                self.live_word_details_overlay.raise_()
            
    def closeEvent(self, event):
        if hasattr(self, 'video_processing_thread') and self.video_processing_thread is not None:
            self.video_processing_thread.stop()
        if hasattr(self, 'live_transcription_thread') and self.live_transcription_thread is not None:
            self.live_transcription_thread.stop()
        if hasattr(self, 'background_video_thread') and self.background_video_thread is not None:
            self.background_video_thread.stop()
        event.accept()

if __name__ == "__main__":
    application = QApplication(sys.argv)
    main_window = SoyleAI()
    main_window.show()
    sys.exit(application.exec())