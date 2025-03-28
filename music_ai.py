import streamlit as st
import librosa
import librosa.display
import google.generativeai as genai
import numpy as np
import pretty_midi
import soundfile as sf
import matplotlib.pyplot as plt
import tempfile
import os

# 🔑 Configure Gemini API
GEMINI_API_KEY = "your_gemini_api_key_here"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# 🎵 Streamlit App Title
st.title("🎶 AI-Generated BGM from Indian Classical Music")
st.markdown("Upload an Indian classical music piece and generate a new BGM using AI!")

# 📤 File Uploader for Audio
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.getvalue())
        audio_path = temp_audio.name

    # 🎵 Load Music File
    y, sr = librosa.load(audio_path)

    # 🎼 Extract Pitch & Tempo Features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = [np.max(pitches[:, i]) for i in range(pitches.shape[1]) if np.max(pitches[:, i]) > 0]
    avg_pitch = np.mean(pitch_values) if pitch_values else 0

    # 📊 Display Tempo & Pitch Information
    st.subheader("🎼 Music Analysis")
    st.write(f"**Tempo:** {tempo:.2f} BPM")
    st.write(f"**Average Pitch:** {avg_pitch:.2f} Hz")

    # 🎻 User Input for AI-Generated BGM
    raga_choice = st.text_input("🎵 Enter Raga Name (e.g., Yaman, Bhairav)", "Yaman")
    mood_choice = st.selectbox("🎭 Choose Mood", ["Peaceful", "Energetic", "Sad", "Meditative", "Joyful"])

    # 🔘 Generate BGM Button
    if st.button("🎶 Generate AI-Based BGM"):
        with st.spinner("Generating music composition... 🎼"):
            # 🧠 AI Prompt for BGM Composition
            prompt = f"""
            I am analyzing an Indian classical music piece with an average pitch of {avg_pitch:.2f} Hz and a tempo of {tempo:.2f} BPM.
            The user wants to generate a new background music (BGM) based on the raga {raga_choice} with a {mood_choice} mood.
            Suggest a melody structure, note sequences, and rhythmic pattern for a new composition.
            """

            # 🔥 Get AI Response
            response = genai.chat(prompt)
            ai_suggestions = response.text
            st.subheader("🎶 AI-Generated Composition")
            st.text(ai_suggestions)

            # 🎼 Convert AI Suggestions to MIDI
            midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)  # 0 = Acoustic Grand Piano
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

            # 📝 Save MIDI File
            midi_path = "generated_bgm.mid"
            midi.write(midi_path)

            # 🔊 Convert MIDI to WAV
            audio_output = "generated_bgm.wav"
            synthesized_audio = midi.synthesize(fs=sr)
            sf.write(audio_output, synthesized_audio, sr)

            st.success("✅ New BGM Generated Successfully!")

            # 📥 Download Links
            with open(midi_path, "rb") as f:
                st.download_button("🎼 Download MIDI", f, file_name="generated_bgm.mid", mime="audio/midi")

            with open(audio_output, "rb") as f:
                st.download_button("🔊 Download Audio (WAV)", f, file_name="generated_bgm.wav", mime="audio/wav")

            # 🎧 Audio Player
            st.audio(audio_output, format="audio/wav")

            # 📊 Pitch Contour Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(pitch_values, label="Pitch Contour", color="purple")
            ax.set_title(f"Pitch Contour for {raga_choice} BGM")
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency (Hz)")
            ax.legend()
            st.pyplot(fig)

    # 🗑️ Clean up temp files
    os.remove(audio_path)
