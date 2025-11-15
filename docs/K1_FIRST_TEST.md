# K-1 First Test - Voice & Autonomous Interaction

**Research & Development Log**
**Developers:** Alan Helmick, Armin Foroughi
**Platform:** K-1 Booster Humanoid (Jetson Orin NX)
**Objective:** Test F5-TTS neural voice, autonomous face exploration, and movement

---

## Pre-Test Setup

### F5-TTS Voice Options

**Size-Appropriate Voice Selection:**

K-1 is 95cm tall (child-sized). Voice pitch should match physical size:
- Smaller bodies = naturally higher voice pitch
- Adult voices from smaller-statured people work great
- Pitch range: 180-250 Hz (vs typical adult male 85-180 Hz)

**Option 1: Use Pre-Recorded Small-Statured Voice (Recommended)**

```bash
# SSH to K-1
ssh booster@192.168.x.x

# Create voice directory
sudo mkdir -p /opt/whoami/voices

# Use voice from:
# - Smaller-statured adults (naturally higher pitch)
# - Teens (13-17 years)
# - Public domain voice samples
# - F5-TTS demo voices

# Download or copy sample:
cd /opt/whoami/voices
# (place your 10-sec WAV file here)
mv your_sample.wav k1_voice.wav
```

**Option 2: Pitch-Shift Any Voice**

Make any voice size-appropriate:

```bash
# Install sox
sudo apt-get install sox

# Record base voice (10 sec)
arecord -D hw:2,0 -f S16_LE -r 48000 -c 2 -d 10 /tmp/base_voice.wav

# Pitch up by 300-500 cents (3-5 semitones)
# 300 = slight higher pitch (smaller adult)
# 400 = noticeably higher (petite adult)
# 500 = very high pitch (very small stature)
sox /tmp/base_voice.wav /opt/whoami/voices/k1_voice.wav pitch 400

# Reference text (have speaker say this):
# "Hello! I'm K-1, a friendly robot helper."
```

**Option 3: Text-to-Speech Generated Sample**

Use another TTS to generate reference:

```bash
# Use espeak to generate base voice with higher pitch
espeak "Hello, I am K-1, a friendly robot helper" -w /tmp/base.wav -s 180 -p 70

# Pitch shift to match small body
sox /tmp/base.wav /opt/whoami/voices/k1_voice.wav pitch 300
```

**Update Config:**

```bash
nano /home/user/whoami/config/k1_booster_config.json

# Update lines 97-98:
"ref_audio": "/opt/whoami/voices/k1_voice.wav",
"ref_text": "Hello! I'm K-1, a friendly robot helper.",
```

---

## Test Protocol

### Test 1: Power Up & Basic Control

**Objective:** Verify robot modes and basic movement

**Steps:**

1. **Power On**
   ```
   - Press K-1 power button
   - Wait 30 seconds for boot
   - Robot initializes in DAMP mode
   - IMPORTANT: Keep robot stationary during boot (IMU calibration)
   ```

2. **SSH Connection**
   ```bash
   ssh booster@192.168.x.x
   cd /home/user/whoami
   ```

3. **Launch Control**
   ```bash
   python basic_controls.py 127.0.0.1
   ```

4. **Mode Sequence**
   ```
   Type: mp    # PREP mode - robot stands up
   Wait 5 sec
   Type: mw    # WALK mode - active control enabled
   ```

5. **Movement Test**
   ```
   w - Forward (2 seconds)
   s - Stop (release keys)
   a - Left strafe (2 seconds)
   d - Right strafe (2 seconds)
   q - Rotate left (2 seconds)
   e - Rotate right (2 seconds)
   ```

6. **Head Control Test**
   ```
   ho - Head origin (center)
   hl - Head left
   hr - Head right
   hu - Head up
   hd - Head down
   ho - Return to center
   ```

**Expected Results:**
- ✅ Robot stands smoothly in PREP mode
- ✅ All movement commands execute
- ✅ Head moves to all positions
- ✅ No joint errors or timeouts

---

### Test 2: Vision System

**Objective:** Verify camera and face detection

**Steps:**

1. **Launch Camera Feed**
   ```bash
   # In new SSH session
   python basic_cam.py
   ```

2. **Access Web Interface**
   ```
   Open browser: http://192.168.x.x:8080
   ```

3. **Launch YOLO Face Detection**
   ```bash
   # Stop basic_cam.py (Ctrl+C)
   python cam_yolo.py --detection face --port 8080
   ```

4. **Face Detection Test**
   ```
   - Stand in front of K-1 (2-3 meters)
   - Move around frame
   - Verify bounding boxes appear
   - Check FPS counter
   ```

**Expected Results:**
- ✅ Camera feed displays in browser
- ✅ Faces detected with bounding boxes
- ✅ Detection runs at 10-15 FPS minimum
- ✅ Multiple faces tracked simultaneously

---

### Test 3: F5-TTS Voice System

**Objective:** Test neural voice synthesis

**Steps:**

1. **Install F5-TTS** (if not installed)
   ```bash
   pip install f5-tts torch torchaudio
   ```

2. **Verify Voice File**
   ```bash
   ls -lh /opt/whoami/voices/k1_voice.wav

   # Should show file size ~500KB-2MB for 10 sec audio
   ```

3. **Test Voice**
   ```bash
   python examples/f5tts_voice_demo.py \
     --ref-audio /opt/whoami/voices/k1_voice.wav \
     --ref-text "Hello! I'm K-1, a friendly robot helper."
   ```

4. **Listen to Output**
   ```
   - Script will speak 3 test phrases
   - Verify size-appropriate voice tone (higher pitch for 95cm body)
   - Check for clarity and naturalness
   ```

**Expected Results:**
- ✅ Voice sounds natural (not robotic)
- ✅ Size-appropriate pitch (matches small stature)
- ✅ Clear pronunciation
- ✅ Generation time < 3 seconds per phrase

---

### Test 4: Autonomous Face Exploration

**Objective:** Test autonomous scanning and person recognition

**Preparation:**

1. **Create Test Profile**
   ```bash
   python -c "
   from whoami.k1_face_explorer import K1FaceExplorer
   explorer = K1FaceExplorer()
   explorer.add_conversation_note('TestPerson', 'robots', 'Interested in K-1 development')
   "
   ```

2. **Launch Autonomous System**
   ```bash
   python examples/k1_autonomous_face_interaction.py 127.0.0.1
   ```

3. **Select Demo Mode**
   ```
   Choose: 1 (Demo interactions)
   ```

4. **Observe Output**
   ```
   Watch console for:
   - Greeting generation
   - Conversation recall
   - Profile updates
   ```

**Expected Results:**
- ✅ Greetings generated with time stamps
- ✅ Conversation notes recalled
- ✅ Profile data persists to JSON
- ✅ System handles multiple people

---

### Test 5: Dance Moves (Bonus)

**Objective:** Test coordinated movement sequences

**Safety First:**
- Clear 2-meter radius around K-1
- Remove obstacles
- Have emergency stop ready (press ESC in control script)

**Simple Dance Sequence:**

```python
# In Python REPL or script
from time import sleep

# Ensure in WALK mode (mw)

# Dance move 1: Side-to-side
for _ in range(3):
    # Left strafe
    send_command('a')  # Hold 1 sec
    sleep(1)
    stop()
    sleep(0.5)

    # Right strafe
    send_command('d')  # Hold 1 sec
    sleep(1)
    stop()
    sleep(0.5)

# Dance move 2: Spin
send_command('q')  # Rotate left
sleep(2)
stop()
sleep(0.5)

send_command('e')  # Rotate right
sleep(2)
stop()

# Dance move 3: Head shake
send_command('hl')  # Left
sleep(0.5)
send_command('hr')  # Right
sleep(0.5)
send_command('ho')  # Center
```

**Note:** For actual implementation, we'll create a proper dance sequence script.

---

## Data Collection

### Metrics to Record

**Movement:**
- [ ] Mode transition times (DAMP→PREP→WALK)
- [ ] Response latency (command to action)
- [ ] Movement accuracy
- [ ] Balance stability

**Vision:**
- [ ] Face detection FPS
- [ ] Detection accuracy (distance range)
- [ ] Multiple face handling
- [ ] Tracking reliability

**Voice:**
- [ ] F5-TTS generation time per phrase
- [ ] Audio clarity (subjective 1-10)
- [ ] Voice appropriateness for child-sized robot
- [ ] Conversation recall accuracy

**Autonomous:**
- [ ] Head scan pattern completion time
- [ ] Face discovery rate
- [ ] Greeting cooldown behavior
- [ ] Profile persistence across sessions

---

## Known Issues & Workarounds

**Issue 1: F5-TTS "No reference audio" error**
```bash
# Check file exists
ls -lh /opt/whoami/voices/k1_voice.wav

# Verify config path matches
cat config/k1_booster_config.json | grep ref_audio
```

**Issue 2: Head movement not responding**
```bash
# Check robot mode - must be PREP or WALK
# Type in control script: gm (get mode)
# If DAMP, transition: mp then mw
```

**Issue 3: Camera feed black screen**
```bash
# Check camera is detected (Zod camera)
v4l2-ctl --list-devices
# Should show: Zod (usb-...) /dev/video0

# If camera not found, replug USB or reboot
sudo reboot
```

---

## Next Steps

After successful testing:

1. **Optimize Voice:**
   - Fine-tune pitch shift (test 300, 400, 500 cents)
   - Try voices from smaller-statured adults
   - Test different reference samples (male vs female pitch ranges)
   - Adjust speaking rate in config

2. **Dance Choreography:**
   - Create dance sequence module
   - Sync with music playback
   - Test multi-move sequences

3. **Conversation Enhancement:**
   - Add more conversation topics
   - Implement topic suggestions
   - Test long-term memory

4. **Multi-Robot Coordination:**
   - Test Gun.js profile sharing
   - Coordinate multiple K-1 units
   - Implement distributed face database

---

## Research Notes

**Voice Cloning Quality Factors:**
- Reference audio quality (background noise affects output)
- Reference text accuracy (must match what's spoken)
- Sample length (10 sec minimum, 30 sec ideal)
- Voice pitch matching robot size (higher pitch for 95cm K-1)
- Natural speaking voice samples work best (avoid shouting/whispering)

**Autonomous Behavior Observations:**
- Head scan pattern efficiency
- Face detection reliability by distance
- Conversation memory retrieval latency
- Person re-identification accuracy

---

**Test Log:** (Update after each test)

| Date | Tester | Test # | Result | Notes |
|------|--------|--------|--------|-------|
|      |        |        |        |       |

---

## Safety Reminders

- ⚠️ Keep 2-meter clear zone during movement tests
- ⚠️ Robot must be stationary during boot (IMU calibration)
- ⚠️ Never lift robot in WALK mode (active balance will resist)
- ⚠️ Have emergency stop ready (ESC key or power button 6-sec hold)
- ⚠️ Monitor battery level (voice will announce low battery)

---

**Document Version:** 1.0
**Last Updated:** 2025-01-15
**Contributors:** Alan Helmick, Armin Foroughi
