import librosa
import librosa.display
import google.generativeai as genai
import numpy as np
import pretty_midi
import soundfile as sf
import matplotlib.pyplot as plt

# 🔑 Configure Gemini API
GEMINI_API_KEY = "your_gemini_api_key_here"
genai.configure(api_key=GEMINI_API_KEY)

# 🎵 Load an Indian classical music piece
audio_file = "path_to_your_stored_music.wav"
y, sr = librosa.load(audio_file)

# 🎼 Extract Pitch, Tempo, and Rhythm Features
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
pitch_values = [np.max(pitches[:, i]) for i in range(pitches.shape[1]) if np.max(pitches[:, i]) > 0]
avg_pitch = np.mean(pitch_values) if pitch_values else 0

# 🎨 User Input for BGM Generation
raga_choice = input("Enter Raga Name (e.g., Yaman, Bhairav): ")
mood_choice = input("Enter Mood (e.g., Peaceful, Energetic, Sad): ")

# 🧠 Use Gemini AI for AI-Driven Music Suggestions
prompt = f"""
I am analyzing an Indian classical music piece with an average pitch of {avg_pitch:.2f} Hz and a tempo of {tempo:.2f} BPM.
The user wants to generate a new background music (BGM) based on the raga {raga_choice} with a {mood_choice} mood.
Suggest a melody structure, note sequences, and rhythmic pattern for a new composition.
"""

response = genai.chat(prompt)
ai_suggestions = response.text
print("\n🎶 AI Suggested Composition:\n", ai_suggestions)

# 🎹 Convert AI Suggestions to MIDI for New BGM
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)  # 0 = Acoustic Grand Piano

# 🎼 Generate MIDI Notes
note_durations = np.linspace(0.5, 1.5, num=len(pitch_values))  # Varying note durations
start_time = 0.0

for pitch, duration in zip(pitch_values, note_durations):
    note = pretty_midi.Note(
        velocity=100,
        pitch=int(pitch % 128),  # MIDI note range (0-127)
        start=start_time,
        end=start_time + duration,
    )
    instrument.notes.append(note)
    start_time += duration

midi.instruments.append(instrument)

# 📝 Save the Generated MIDI
midi_path = "generated_bgm.mid"
midi.write(midi_path)
print(f"🎼 New BGM saved as: {midi_path}")

# 🔊 Convert MIDI to Audio (WAV)
audio_output = "generated_bgm.wav"
synthesized_audio = midi.synthesize(fs=sr)
sf.write(audio_output, synthesized_audio, sr)
print(f"🔊 Audio saved as: {audio_output}")

# 📊 Plot Pitch Contour
plt.figure(figsize=(10, 4))
plt.plot(pitch_values, label="Pitch Contour", color="purple")
plt.title(f"Pitch Contour for {raga_choice} BGM")
plt.xlabel("Time")
plt.ylabel("Frequency (Hz)")
plt.legend()
plt.show()
