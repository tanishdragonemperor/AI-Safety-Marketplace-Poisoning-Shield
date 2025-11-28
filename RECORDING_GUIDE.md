# 🎥 VIDEO RECORDING GUIDE
## Marketplace Poisoning Shield - Full Explained Demo

---

## 📋 BEFORE RECORDING

### Setup Checklist:
- [ ] Open terminal in full screen
- [ ] Increase font size (Ctrl + or Cmd +) for visibility
- [ ] Navigate to project folder: `cd marketplace-poisoning-shield`
- [ ] Activate venv: `source venv/bin/activate`
- [ ] Test the demo first: `python video_demo.py`
- [ ] Close unnecessary apps
- [ ] Start screen recording software

---

## 🌐 FRONTEND DEMO (RECOMMENDED FOR VIDEO!)

The easiest way to demo is using the **web frontend**:

```bash
# Make sure venv is activated
source venv/bin/activate

# Start the server (opens browser automatically)
python run_server.py
```

This gives you a beautiful UI with:
- 📊 Dashboard with live stats
- ⚔️ Red Team attack simulator
- 🛡️ Blue Team defense visualization
- 🔍 Search comparison demo
- 📦 Product listing with threat scores

### Frontend Recording Tips:
1. Click through each tab while explaining
2. Run attack simulations live
3. Show search comparisons
4. Point at the metrics and graphs

### Recording Settings:
- Resolution: 1920x1080 recommended
- Record microphone audio
- Record system audio (optional)

---

## 🎬 RECORDING SCRIPT

### INTRO (0:00 - 1:00)
**Run:** `python video_demo.py`

**SAY:**
> "Hi, I'm Tanish Gupta, and this is my AI Security project called Marketplace Poisoning Shield."
>
> "The problem I'm addressing is something called Silent Data Poisoning - a type of attack that targets AI-powered e-commerce platforms like Amazon, eBay, and Alibaba."
>
> "These platforms use AI for search ranking, recommendations, and fraud detection. But here's the vulnerability: these AI systems learn from user-provided data."
>
> "An attacker can inject poisoned data that manipulates the AI - making their products rank higher, boosting fake ratings, or evading moderation."
>
> "What makes these attacks dangerous is they're SILENT - they're invisible to human moderators."

**[Press ENTER to continue]**

---

### PART 1: THE THREAT (1:00 - 3:00)

**SAY:**
> "Let me show you exactly how these invisible attacks work with a real example."
>
> "Look at these two product titles on screen. They look completely identical, right?"
>
> "But watch what happens when we check the byte count..."
>
> "The clean text is 27 bytes, but the poisoned text is 33 bytes. There are 6 hidden bytes!"
>
> "These are called ZERO-WIDTH CHARACTERS. They're Unicode characters that have no visual representation but exist in the data."
>
> "When the AI tokenizer processes this text, it sees these hidden characters and creates different embeddings. This can manipulate search rankings."
>
> "Another attack is called HOMOGLYPH ATTACK. Here we replace characters with visually identical ones from different alphabets."
>
> "For example, the Cyrillic letter 'е' looks exactly like the Latin 'e', but they have different Unicode values. This tricks AI systems that rely on exact matching."

**[Press ENTER to continue]**

---

### PART 2: DATASET (3:00 - 4:00)

**SAY:**
> "For this project, I built a synthetic e-commerce dataset generator."
>
> "It creates realistic product listings with titles, descriptions, prices, ratings, reviews, and metadata."
>
> "I generated 100 products across 5 categories: electronics, clothing, home, sports, and beauty."
>
> "Each product has all the attributes you'd find on a real marketplace - seller information, customer reviews, keywords, and more."
>
> "This gives us a realistic testing environment for our attacks and defenses."

**[Press ENTER to continue]**

---

### PART 3: ATTACKS - RED TEAM (4:00 - 8:00)

**SAY:**
> "Now let's look at the RED TEAM side - the attack simulation."
>
> "I implemented 6 different types of poisoning attacks. Let me demonstrate each one."

**Attack 1 - Hidden Characters:**
> "Attack 1 is HIDDEN CHARACTER INJECTION. Watch the byte count change."
>
> "We go from [X] bytes to [Y] bytes. Those extra bytes are invisible zero-width characters."
>
> "A human looking at this listing would see nothing wrong. But the AI processes it differently."

**[Press ENTER]**

**Attack 2 - Keyword Stuffing:**
> "Attack 2 is KEYWORD STUFFING - a classic SEO spam technique."
>
> "We inject high-value keywords like 'bestseller', 'top rated', 'premium' into the description."
>
> "This artificially boosts the product's relevance score in search results."

**[Press ENTER]**

**Attack 3 - Fake Reviews:**
> "Attack 3 is FAKE REVIEW INJECTION."
>
> "Watch the rating change - from 4.2 stars to 4.8 stars."
>
> "We inject fake 5-star reviews with generic positive text like 'AMAZING! Best product ever!'"
>
> "This is exactly what fraudulent sellers do on real platforms."

**[Press ENTER]**

**Attack 4 - Homoglyphs:**
> "Attack 4 is the HOMOGLYPH ATTACK I mentioned earlier."
>
> "We replace Latin characters with Cyrillic or Greek lookalikes."
>
> "The text LOOKS identical, but it has different character codes."
>
> "This can bypass keyword filters and brand protection systems."

**[Press ENTER]**

**Dataset Poisoning:**
> "Now I'll poison the entire dataset with a 25% poison rate."
>
> "You can see the distribution of attacks - we have a mix of all 6 types."
>
> "This simulates a realistic attack scenario where multiple attackers use different techniques."

**[Press ENTER]**

---

### PART 4: DEFENSE - BLUE TEAM (8:00 - 11:00)

**SAY:**
> "Now let's look at the BLUE TEAM side - the defense system."
>
> "I implemented a MULTI-LAYER defense architecture with 6 detection layers."
>
> "This is called 'Defense in Depth' - the idea is that no single defense catches everything, but together they provide robust protection."

**Explain each layer:**
> "Layer 1: UNICODE ANOMALY DETECTION - scans for hidden and zero-width characters."
>
> "Layer 2: KEYWORD DENSITY ANALYSIS - detects SEO spam patterns and unusual word repetition."
>
> "Layer 3: HOMOGLYPH DETECTION - I built a mapping of 50+ known character substitutions."
>
> "Layer 4: REVIEW AUTHENTICITY CHECKER - uses pattern matching to identify fake reviews."
>
> "Layer 5: METADATA VALIDATION - checks for hidden fields and implausible metrics."
>
> "Layer 6: STATISTICAL ANOMALY DETECTION - compares products against a learned baseline."

**[Press ENTER]**

> "Let me run the defense analysis on our poisoned dataset..."
>
> "You can see the results - we detected [X] suspicious products out of [Y]."
>
> "Each product gets a THREAT SCORE from 0 to 1. Higher scores indicate more suspicious products."
>
> "Look at these sample detections - the system correctly identified different attack types."

**[Press ENTER]**

---

### PART 5: SEARCH IMPACT (11:00 - 14:00)

**SAY:**
> "This is the KEY INSIGHT of the project - how does all this affect REAL USERS?"
>
> "I built a semantic search pipeline using embeddings and vector similarity."
>
> "This simulates how real marketplace search works - converting queries and products to vectors, then finding the most similar ones."
>
> "Let me show you the difference WITH and WITHOUT defense..."

**For each search query:**
> "Search query: '[query]'"
>
> "WITHOUT DEFENSE: Look at the results. [X] out of 5 top results are POISONED products."
>
> "These are fraudulent listings that gamed their way to the top!"
>
> "WITH DEFENSE: Now look at the protected results. Only [Y] poisoned products."
>
> "The defense blocked [Z] poisoned products from reaching the top results."
>
> "This is the real-world impact - users see genuine products instead of fraudulent ones."

**[Press ENTER between searches]**

---

### PART 6: EVALUATION (14:00 - 16:00)

**SAY:**
> "Finally, let's look at the formal EVALUATION METRICS."
>
> "I use standard machine learning classification metrics."

**Explain the confusion matrix:**
> "Here's the confusion matrix:"
>
> "TRUE POSITIVES: [X] - Poisoned products we correctly detected."
>
> "FALSE POSITIVES: [X] - Clean products we incorrectly flagged."
>
> "FALSE NEGATIVES: [X] - Poisoned products we missed."
>
> "TRUE NEGATIVES: [X] - Clean products we correctly allowed."

**Explain metrics:**
> "PRECISION is [X]% - of the products we flagged, [X]% were actually poisoned. Low false alarms!"
>
> "RECALL is [X]% - of the poisoned products, we caught [X]%. High detection rate!"
>
> "F1 SCORE is [X]% - the harmonic mean, showing overall effectiveness."
>
> "Looking at per-attack detection rates, you can see which attacks are easier or harder to detect."
>
> "Hidden characters and homoglyphs have very high detection rates."
>
> "Adversarial paraphrasing is harder to detect because it's semantically valid text."

**[Press ENTER]**

---

### CONCLUSION (16:00 - 17:00)

**SAY:**
> "To wrap up, here are the key takeaways:"
>
> "First, Silent Data Poisoning is a REAL THREAT. AI systems that learn from user data are vulnerable."
>
> "Second, MULTI-LAYER DEFENSE is essential. No single technique catches everything."
>
> "Third, we achieved MEASURABLE RESULTS - over 90% F1 score with low false positives."
>
> "Fourth, there's PRACTICAL IMPACT - our defense protects search results from manipulation."
>
> "For future work, I'd like to explore neural embedding-based detection using transformers, and real-time streaming defense for production systems."
>
> "Thank you for watching! This has been Marketplace Poisoning Shield for Intro to AI Security."

---

## 🎬 AFTER RECORDING

### Post-Production Checklist:
- [ ] Trim intro/outro dead air
- [ ] Add title card at beginning
- [ ] Add your name and course info
- [ ] Check audio levels
- [ ] Export in required format

### Recommended Video Length:
- Short version: 8-10 minutes (skip some attacks)
- Full version: 15-17 minutes (all content)

---

## 🆘 TROUBLESHOOTING

**If demo crashes:**
```bash
python -c "import sys; sys.path.insert(0,'src'); from dataset_generator import *; print('OK')"
```

**If you need to restart:**
```bash
python video_demo.py
```

**Quick test run:**
```bash
python main.py --dataset-size 50
```

---

## 📁 FILES OVERVIEW

| File | Purpose |
|------|---------|
| `video_demo.py` | Main recording script |
| `main.py` | Full demo with all features |
| `presentation_demo.py` | Classroom presentation |
| `src/attack_simulator.py` | Red team attacks |
| `src/defense_module.py` | Blue team defenses |
| `src/search_pipeline.py` | Search system |
| `src/evaluation.py` | Metrics calculation |

Good luck with your recording! 🎥
