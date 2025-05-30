# 🧠 ML API with FastAPI

This project provides a RESTful API for serving collaboration quality prediction models using FastAPI. It includes all the necessary components for data processing, model training, evaluation, and deployment.

---

## 📁 Project Structure

```
.
├── ReadMe.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── data/
│   └── processed/              # Processed data used for training
├── models/
│   ├── trained_models/         # Serialized trained ML models
│   └── model_evaluation/       # Evaluation metrics and visualization files
├── notebooks/                  # Jupyter notebooks for exploration and development
├── src/
│   ├── api/
│   │   └── main_api.py         # FastAPI main entrypoint
│   └── training/
│       └── build_model.py      # Script to train ML models
├── tests/                      # Unit tests and test scripts
└── venv/                       # Python virtual environment (optional, local only)
```

---

## 📦 Installation

To set up and run the project locally:

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **(Optional) Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate     # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the FastAPI Server

To start the FastAPI server, run the following command from the root directory:

```bash
uvicorn src.api.main_api:app --reload
```

- Open your browser and navigate to: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🛠 Development Notes

- **Training:**  
  Run `build_model.py` inside `src/training/` to train models using data from `data/processed/`.

- **Models:**  
  Trained models are stored in `models/trained_models/`. Evaluation results (metrics, plots) are in `models/model_evaluation/`.

- **Testing:**  
  Place unit and integration tests inside the `tests/` directory.

- **Notebooks:**  
  Use `notebooks/` for data exploration, prototyping, and analysis.

---

## 🧾 License

Add your preferred license here (e.g., MIT, Apache 2.0).

---

## 🙋‍♂️ Author

Developed by [Your Name]
