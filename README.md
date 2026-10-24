# sarvam_hackathon

## Disha v2 — Clinical Voice AI Agent Powered by Sarvam AI

An intelligent, clinically safe voice companion for women's wellness in India, integrating:
- **Sarvam Saaras v3 STT**: Indic Speech-to-Text with transliterated Hinglish recognition.
- **Disha v2 Parallel Fast-Path**:
  - Fine-tuned **MiniLM** Cardiac/Neuro Emergency Classifier + Aho-Corasick backstop (<20ms).
  - Fine-tuned **Laya v2** Multi-Task Intent & Tool Router.
  - SentenceTransformer + Redis Knowledge Card Retriever.
  - Zero-Tolerance Output Guardrail (<0.5ms).
- **Sarvam Bulbul v3 TTS**: High-fidelity Indian voice synthesis (Ritu, Priya, Roopa).
- **Interactive Live Voice Call Web Suite**: Browser microphone recording, live audio visualizer, automatic voice turn playback, and real-time latency waterfall breakdown.

---

### Setup & Quickstart

1. Install dependencies:
   ```bash
   pip install sarvamai python-dotenv
   ```
2. Configure `.env`:
   ```bash
   SARVAM_API_KEY=your_sarvam_api_key_here
   ```
3. Run the Interactive Voice Call Web Suite:
   ```bash
   python web_demo.py
   ```
   Open `http://localhost:8000` in your browser.

4. Run the CLI Voice Demo:
   ```bash
   python run_sarvam_voice_demo.py
   ```
