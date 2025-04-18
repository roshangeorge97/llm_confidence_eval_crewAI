# Ground-Truth-Guided Self-Correction in LLMs

This project implements a Ground-Truth-Guided Evaluation (GTGE) framework using a multi-agent Crew AI system to improve the self-correction capabilities of Large Language Models (LLMs).

## 🚀 Overview
LLMs often generate plausible but incorrect responses and lack robust self-correction abilities. This project introduces a novel multi-agent system using Crew AI to:

- Evaluate LLM responses against ground-truth answers
- Refine incorrect outputs using a Refiner Agent
- Improve overall accuracy across various benchmarks

## 🧠 Key Features
- **Crew AI Integration**: Multi-agent architecture with Evaluator and Refiner agents
- **Ground-Truth Benchmarking**: Uses known correct answers to guide corrections
- **Single-Pass Correction**: Enhances efficiency without iterative loops
- **Support for Gemini Models**: Tested on Gemini 2.0 Flash, Flash-Lite, 1.5 Flash, 1.5 Flash-8B, and 1.5 Pro

## 📊 Benchmarks
Evaluated on six diverse tasks:
- GSM8K (Math Reasoning)
- SVAMP (Arithmetic)
- HotpotQA (Multi-hop QA)
- Sports (Commonsense)
- LLC (Symbolic Reasoning)
- Domestic Robot (Instruction Following)

## 📈 Results
The GTGE framework demonstrated the following improvements:

| Model             | Dataset   | Baseline Accuracy | GTGE Accuracy |
|------------------|-----------|-------------------|----------------|
| Gemini 1.5 Pro   | GSM8K     | 87.0%             | **96.0%**      |
| Gemini 2.0 Flash | SVAMP     | 86.0%             | **94.5%**      |
| Gemini 1.5 Flash | HotpotQA  | 66.0%             | **73.0%**      |
| Gemini 2.0 F-Lite| Sports    | 74.0%             | **82.0%**      |
| Gemini 1.5 Pro   | LLC       | 64.0%             | **96.0%**      |

- Up to **12% improvement** in accuracy
- Higher correction and consistency rates compared to baseline and confidence-based approaches
- Stronger correlation between confidence and correctness in structured tasks

## 🛠️ Technologies Used
- Python
- Crew AI
- Gemini API (via prompt-based LLM interface)
- Matplotlib, Pandas, NumPy

## 📁 Structure
```
├── agents/                  # Crew AI agent definitions
├── benchmarks/              # Datasets and task definitions
├── core/                    # GTGE evaluation and refinement logic
├── results/                 # Plots, accuracy logs, visualizations
└── calm_core.py                  # Entry point
```

## 🧪 How to Run
1. Clone the repository
2. Install requirements
```bash
pip install -r requirements.txt
```
3. Run the main script
```bash
python calm_core.py
```

## 📬 Contact
For questions or collaborations, reach out via [LinkedIn](https://www.linkedin.com/in/roshangeorge97) or check the [GitHub repository](https://github.com/roshangeorge97/llm_confidence_eval_crewAI).

---

Made with ❤️ by Roshan George
