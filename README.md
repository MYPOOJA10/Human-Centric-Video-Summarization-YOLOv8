# Human-Centric Video Summarization using YOLOv8

A human-centric video summarization framework for smart surveillance that uses **YOLOv8-based human detection, motion analysis, and event importance evaluation** to identify meaningful activities and reduce redundant video storage.

## Project Overview

The rapid deployment of surveillance cameras generates large volumes of continuous video data. Storing all recorded footage requires significant storage capacity and makes important events difficult to retrieve.

This project proposes an intelligent surveillance video summarization framework that focuses on **human presence and activities**. The system analyzes surveillance video at predefined intervals, detects humans using YOLOv8, evaluates motion and event importance, and makes intelligent storage decisions.

Important events are preserved, while less significant or inactive video segments can be compressed or discarded to reduce storage requirements.

## Objectives

* Detect humans in surveillance videos using YOLOv8.
* Analyze motion to identify significant activities.
* Calculate an Event Importance Score for storage decisions.
* Preserve important surveillance events.
* Reduce redundant video storage.
* Generate lightweight metadata for efficient event retrieval.
* Provide a dashboard for monitoring summarized events and storage statistics.

## Key Features

* **Human Detection:** YOLOv8-based real-time human detection.
* **Motion Analysis:** Identifies activity levels within video segments.
* **Event Importance Evaluation:** Combines human presence, motion, and event duration.
* **Intelligent Storage:** Classifies video segments for storage, compression, or discarding.
* **Metadata Generation:** Stores timestamps, human counts, motion scores, and event information.
* **Video Summarization:** Retains meaningful portions of surveillance footage.
* **Dashboard:** Provides visual information about detected events and storage optimization.
* **Automated Processing:** Supports automated video analysis and event processing.

## Technology Stack

* **Python**
* **YOLOv8**
* **OpenCV**
* **Flask**
* **HTML/CSS**
* **Pandas**
* **NumPy**
* **Computer Vision**
* **Motion Analysis**

## System Workflow

```text
Input Surveillance Video
          ↓
Video Preprocessing
          ↓
YOLOv8 Human Detection
          ↓
Human Count + Motion Analysis
          ↓
Event Importance Score
          ↓
Event Classification
          ↓
 ┌────────┼───────────┐
 ↓        ↓           ↓
Store   Compress    Discard
          ↓
Metadata Generation
          ↓
Video Summary + Dashboard
```

## Event Importance Score

The framework evaluates the importance of a video segment using multiple factors:

* **Human Count (HC)**
* **Motion Score (MS)**
* **Event Duration (ED)**

These factors are combined to determine the significance of an event and guide the storage decision.

```text
Event Importance Score
          ↓
   ┌──────┼──────┐
   ↓      ↓      ↓
 Store  Compress Discard
```

## Metadata

The system generates lightweight metadata to support efficient indexing and retrieval.

Example metadata information includes:

* Timestamp
* Number of detected people
* Motion score
* Event duration
* Event label
* Storage decision

## Results

The proposed framework demonstrated effective human detection and intelligent video storage optimization.

Key experimental observations include:

* Human detection using YOLOv8 achieved high detection accuracy.
* The framework achieved approximately **13–18 FPS** processing speed depending on the workload.
* Intelligent event-based storage decisions helped reduce redundant video storage.
* In the evaluated scenario, storage consumption was reduced by approximately **57%**.

## Project Structure

```text
Adaptive_Surveillance_Project_New/
│
├── final_summary/
├── metadata/
│   ├── events_metadata.csv
│   └── session.json
│
├── models/
│   └── yolov8n.pt
│
├── templates/
│   └── index.html
│
├── dashboard.py
├── main_dynamic_storage_updated.py
├── merge_events.py
├── requirements.txt
└── setup_and_run.bat
```

## Installation

Clone the repository:

```bash
git clone https://github.com/MYPOOJA10/Human-Centric-Video-Summarization-YOLOv8.git
cd Human-Centric-Video-Summarization-YOLOv8
```

Install the required Python packages:

```bash
pip install -r Adaptive_Surveillance_Project_New/requirements.txt
```

## Running the Project

Navigate to the project directory:

```bash
cd Adaptive_Surveillance_Project_New
```

Run the required Python application according to the project configuration.

For the dashboard, run:

```bash
python dashboard.py
```

Then open the local Flask URL displayed in the terminal.

## Applications

This framework can be useful in:

* Smart city surveillance
* Educational institutions
* Transportation hubs
* Commercial buildings
* Public-area monitoring
* Security and safety monitoring

## Future Scope

* Integration with additional object detection models.
* Real-time camera stream processing.
* Advanced activity recognition.
* Cloud-based surveillance storage.
* Improved event classification using deep learning.
* Automated alerts for critical events.
* Further optimization of storage and processing performance.

## Author

**Pooja Myakala**

M.Tech – Data Science
B.Tech – Computer Science and Engineering

---

### Note

This project was developed as part of an academic/research project on **Human-Centric Video Summarization with Adaptive Storage Optimization**.

